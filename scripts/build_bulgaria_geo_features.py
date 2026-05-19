#!/usr/bin/env python3
"""Join real Bulgaria postal-code geography onto the Baza churn table."""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.helpers import processed_data_path, raw_data_path, resolve_input_path

GEONAMES_URL = "https://download.geonames.org/export/zip/BG.zip"
SOFIA = (42.6977, 23.3219)
PLOVDIV = (42.1354, 24.7453)
VARNA = (43.2141, 27.9147)
BURGAS = (42.5048, 27.4626)


def normalize_postal_code(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.extract(r"(\d{4})", expand=False)
    )


def english_name(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.split("/", n=1)
        .str[-1]
        .str.strip()
        .replace("", pd.NA)
    )


def mode_or_unknown(values: pd.Series) -> str:
    clean = values.dropna().astype(str)
    if clean.empty:
        return "Unknown"
    return str(clean.mode().iloc[0])


def haversine_km(lat: pd.Series, lon: pd.Series, target: tuple[float, float]) -> pd.Series:
    radius = 6371.0
    lat1 = np.radians(pd.to_numeric(lat, errors="coerce"))
    lon1 = np.radians(pd.to_numeric(lon, errors="coerce"))
    lat2 = math.radians(target[0])
    lon2 = math.radians(target[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * math.cos(lat2) * np.sin(dlon / 2) ** 2
    return radius * 2 * np.arcsin(np.sqrt(a))


def load_geonames_bg(path: Path) -> pd.DataFrame:
    columns = [
        "country_code",
        "postal_code",
        "place_name",
        "admin_name1",
        "admin_code1",
        "admin_name2",
        "admin_code2",
        "admin_name3",
        "admin_code3",
        "latitude",
        "longitude",
        "accuracy",
    ]
    with zipfile.ZipFile(path) as archive:
        with archive.open("BG.txt") as handle:
            geo = pd.read_csv(
                handle,
                sep="\t",
                header=None,
                names=columns,
                dtype=str,
                encoding="utf-8",
            )

    geo["postal_code"] = normalize_postal_code(geo["postal_code"])
    for col in ["place_name", "admin_name1", "admin_name2", "admin_name3"]:
        geo[f"{col}_en"] = english_name(geo[col])
    for col in ["latitude", "longitude", "accuracy"]:
        geo[col] = pd.to_numeric(geo[col], errors="coerce")

    grouped = (
        geo.dropna(subset=["postal_code"])
        .groupby("postal_code", dropna=False)
        .agg(
            bg_place_count=("place_name_en", "size"),
            bg_place_name=("place_name_en", mode_or_unknown),
            bg_admin1_name=("admin_name1_en", mode_or_unknown),
            bg_admin1_code=("admin_code1", mode_or_unknown),
            bg_admin2_name=("admin_name2_en", mode_or_unknown),
            bg_admin2_code=("admin_code2", mode_or_unknown),
            bg_admin3_name=("admin_name3_en", mode_or_unknown),
            bg_admin3_code=("admin_code3", mode_or_unknown),
            bg_latitude=("latitude", "mean"),
            bg_longitude=("longitude", "mean"),
            bg_geo_accuracy=("accuracy", "median"),
        )
        .reset_index()
    )
    return grouped


def enrich_baza(input_csv: Path, geonames_zip: Path, output_csv: Path, metadata_out: Path) -> None:
    baza = pd.read_csv(input_csv)
    baza.columns = baza.columns.str.strip()
    if "Billing_ZIP" not in baza.columns:
        raise ValueError(f"Billing_ZIP column not found: {input_csv}")

    geo = load_geonames_bg(geonames_zip)
    baza["Billing_ZIP_norm"] = normalize_postal_code(baza["Billing_ZIP"])
    enriched = baza.merge(geo, left_on="Billing_ZIP_norm", right_on="postal_code", how="left")
    enriched["bg_zip_found"] = enriched["postal_code"].notna().astype(int)

    for col in ["bg_place_count", "bg_geo_accuracy", "bg_latitude", "bg_longitude"]:
        enriched[col] = pd.to_numeric(enriched[col], errors="coerce")

    for city, coords in [
        ("sofia", SOFIA),
        ("plovdiv", PLOVDIV),
        ("varna", VARNA),
        ("burgas", BURGAS),
    ]:
        enriched[f"bg_distance_{city}_km"] = haversine_km(
            enriched["bg_latitude"], enriched["bg_longitude"], coords
        )

    enriched["bg_min_big_city_distance_km"] = enriched[
        [
            "bg_distance_sofia_km",
            "bg_distance_plovdiv_km",
            "bg_distance_varna_km",
            "bg_distance_burgas_km",
        ]
    ].min(axis=1)

    for col in [
        "bg_place_name",
        "bg_admin1_name",
        "bg_admin1_code",
        "bg_admin2_name",
        "bg_admin2_code",
        "bg_admin3_name",
        "bg_admin3_code",
    ]:
        enriched[col] = enriched[col].fillna("Unknown").astype(str)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_csv, index=False, encoding="utf-8-sig")

    matched_rows = int(enriched["bg_zip_found"].sum())
    metadata = {
        "source": "GeoNames Bulgaria postal codes",
        "source_url": GEONAMES_URL,
        "license_note": "GeoNames postal-code data is distributed under a Creative Commons Attribution license.",
        "input_csv": str(input_csv),
        "geonames_zip": str(geonames_zip),
        "output_csv": str(output_csv),
        "rows": int(len(enriched)),
        "matched_rows": matched_rows,
        "match_rate": float(matched_rows / max(len(enriched), 1)),
        "unique_baza_postal_codes": int(enriched["Billing_ZIP_norm"].nunique(dropna=True)),
        "unique_geonames_postal_codes": int(geo["postal_code"].nunique(dropna=True)),
        "added_columns": [
            "Billing_ZIP_norm",
            "bg_zip_found",
            "bg_place_count",
            "bg_place_name",
            "bg_admin1_name",
            "bg_admin1_code",
            "bg_admin2_name",
            "bg_admin2_code",
            "bg_admin3_name",
            "bg_admin3_code",
            "bg_latitude",
            "bg_longitude",
            "bg_geo_accuracy",
            "bg_distance_sofia_km",
            "bg_distance_plovdiv_km",
            "bg_distance_varna_km",
            "bg_distance_burgas_km",
            "bg_min_big_city_distance_km",
        ],
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[done] rows={len(enriched)} matched={matched_rows} "
        f"match_rate={metadata['match_rate']:.2%} output={output_csv}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=raw_data_path("baza_telecom_v2.csv"))
    parser.add_argument("--geonames-zip", type=Path, default=raw_data_path("BG_postal_codes_geonames.zip"))
    parser.add_argument("--output", type=Path, default=processed_data_path("baza_telecom_v2_bulgaria_geo.csv"))
    parser.add_argument("--metadata-out", type=Path, default=processed_data_path("baza_telecom_v2_bulgaria_geo_metadata.json"))
    args = parser.parse_args()

    args.input = resolve_input_path(args.input, raw_data_path("baza_telecom_v2.csv"))
    args.geonames_zip = resolve_input_path(args.geonames_zip, raw_data_path("BG_postal_codes_geonames.zip"))
    args.output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    args.metadata_out = args.metadata_out if args.metadata_out.is_absolute() else REPO_ROOT / args.metadata_out

    if not args.input.is_file():
        raise SystemExit(f"CSV not found: {args.input}")
    if not args.geonames_zip.is_file():
        raise SystemExit(f"GeoNames ZIP not found: {args.geonames_zip}")

    enrich_baza(args.input, args.geonames_zip, args.output, args.metadata_out)


if __name__ == "__main__":
    main()
