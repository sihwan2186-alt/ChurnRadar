#!/usr/bin/env python3
"""Join official Bulgaria NSI public demographic data onto Baza geography."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.helpers import processed_data_path, raw_data_path, resolve_input_path

NSI_POPULATION_URL = "https://dev.nsi.bg/en/content/2975/population-districts-municipalities-place-residence-and-sex"
NSI_WORKING_AGE_URL = (
    "https://dev.nsi.bg/en/content/13038/"
    "population-working-age-status-place-residence-sex-and-districts-and-municipalities"
)

DISTRICTS = {
    "blagoevgrad",
    "burgas",
    "varna",
    "veliko tarnovo",
    "vidin",
    "vratsa",
    "gabrovo",
    "dobrich",
    "kardzhali",
    "kyustendil",
    "lovech",
    "montana",
    "pazardzhik",
    "pernik",
    "pleven",
    "plovdiv",
    "razgrad",
    "ruse",
    "silistra",
    "sliven",
    "smolyan",
    "sofia",
    "sofia stolitsa",
    "stara zagora",
    "targovishte",
    "haskovo",
    "shumen",
    "yambol",
}

NAME_REPLACEMENTS = {
    "khaskovo": "haskovo",
    "kurdzhali": "kardzhali",
    "kurdjali": "kardzhali",
    "kjustendil": "kyustendil",
    "turgovishhe": "targovishte",
    "vraca": "vratsa",
    "jambol": "yambol",
    "sofija": "sofia",
    "sofija stolica": "sofia stolitsa",
    "smoljan": "smolyan",
    "peshhera": "peshtera",
    "panagjurishhe": "panagyurishte",
    "purvomaj": "parvomay",
    "khisarja": "hisarya",
    "veliko turnovo": "veliko tarnovo",
    "aksakovo": "aksakovo",
    "vulchi dol": "valchi dol",
    "vulchedrum": "valchedram",
    "shhipsko": "shipsko",
    "marica": "maritsa",
    "kazanluk": "kazanlak",
    "kharmanli": "harmanli",
    "stambolijski": "stamboliyski",
    "ljubimec": "lyubimets",
    "suedinenie": "saedinenie",
    "kalojanovo": "kaloyanovo",
    "bracigovo": "bratsigovo",
    "perushhica": "perushtitsa",
    "muglizh": "maglizh",
    "bratja daskalovi": "bratya daskalovi",
    "luki": "laki",
    "gulubovo": "galabovo",
    "carevo": "tsarevo",
    "berkovica": "berkovitsa",
    "ljaskovec": "lyaskovets",
    "zlatica": "zlatitsa",
    "bjala": "byala",
    "pavel banja": "pavel banya",
}


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    flat = []
    for col in frame.columns:
        if isinstance(col, tuple):
            parts = [str(part).strip() for part in col if str(part).strip()]
            flat.append("_".join(parts))
        else:
            flat.append(str(col).strip())
    frame.columns = flat
    return frame


def clean_number(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).replace("\xa0", "").replace(" ", "").strip()
    if text in {"", "-"}:
        return 0.0
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return np.nan


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("А", "A").replace("а", "a")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("(stolitsa)", "stolitsa")
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return NAME_REPLACEMENTS.get(text, text)


def load_population_table(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = flatten_columns(pd.read_html(path)[0])
    table = table.rename(columns={
        table.columns[0]: "area",
        table.columns[1]: "nsi_population_total",
        table.columns[2]: "nsi_population_male",
        table.columns[3]: "nsi_population_female",
        table.columns[4]: "nsi_population_urban",
        table.columns[7]: "nsi_population_rural",
    })
    keep = [
        "area",
        "nsi_population_total",
        "nsi_population_male",
        "nsi_population_female",
        "nsi_population_urban",
        "nsi_population_rural",
    ]
    table = table[keep].copy()
    for col in keep[1:]:
        table[col] = table[col].map(clean_number)

    records = []
    current_district = ""
    for row in table.itertuples(index=False):
        area = str(row.area).strip()
        area_norm = normalize_name(area)
        if not area_norm or area_norm == "total":
            continue
        if area_norm in DISTRICTS and current_district != area_norm:
            current_district = area_norm
            level = "district"
        else:
            level = "municipality"
        record = row._asdict()
        record["area_norm"] = area_norm
        record["district_norm"] = current_district if level == "municipality" else area_norm
        record["level"] = level
        records.append(record)

    parsed = pd.DataFrame(records)
    for col in [
        "nsi_population_male",
        "nsi_population_female",
        "nsi_population_urban",
        "nsi_population_rural",
    ]:
        parsed[f"{col}_ratio"] = parsed[col] / parsed["nsi_population_total"].replace(0, np.nan)

    districts = parsed[parsed["level"] == "district"].drop(columns=["level"]).copy()
    municipalities = parsed[parsed["level"] == "municipality"].drop(columns=["level"]).copy()
    return districts, municipalities


def load_working_age_table(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = flatten_columns(pd.read_html(path)[0])
    table = table.rename(columns={
        table.columns[0]: "area_or_category",
        table.columns[1]: "total",
        table.columns[2]: "male",
        table.columns[3]: "female",
        table.columns[4]: "urban_total",
    })

    age_categories = {
        "under working age": "under",
        "at working age": "working",
        "over working age": "over",
    }
    records_by_key: dict[tuple[str, str], dict[str, float | str]] = {}
    current_area = ""
    current_area_norm = ""
    current_district = ""
    current_level = ""

    for _, row in table.iterrows():
        label = str(row["area_or_category"]).strip()
        label_norm = normalize_name(label)
        if not label_norm or label_norm.startswith("working age in"):
            continue
        category = age_categories.get(label_norm)
        if category is None:
            if label_norm == "total":
                current_area = ""
                current_area_norm = ""
                current_level = ""
                continue
            current_area = label
            current_area_norm = label_norm
            if current_area_norm in DISTRICTS and current_district != current_area_norm:
                current_district = current_area_norm
                current_level = "district"
            else:
                current_level = "municipality"
            key = (current_district if current_level == "municipality" else current_area_norm, current_area_norm)
            records_by_key[key] = {
                "area": current_area,
                "area_norm": current_area_norm,
                "district_norm": key[0],
                "level": current_level,
                "nsi_workage_population_total": clean_number(row["total"]),
            }
            continue

        if not current_area_norm or not current_level:
            continue
        key = (current_district if current_level == "municipality" else current_area_norm, current_area_norm)
        record = records_by_key.setdefault(key, {
            "area": current_area,
            "area_norm": current_area_norm,
            "district_norm": key[0],
            "level": current_level,
        })
        record[f"nsi_{category}_working_age_total"] = clean_number(row["total"])
        record[f"nsi_{category}_working_age_urban"] = clean_number(row["urban_total"])

    parsed = pd.DataFrame(records_by_key.values())
    for col in [
        "nsi_under_working_age_total",
        "nsi_working_working_age_total",
        "nsi_over_working_age_total",
    ]:
        parsed[f"{col}_ratio"] = parsed[col] / parsed["nsi_workage_population_total"].replace(0, np.nan)

    districts = parsed[parsed["level"] == "district"].drop(columns=["level"]).copy()
    municipalities = parsed[parsed["level"] == "municipality"].drop(columns=["level"]).copy()
    return districts, municipalities


def add_prefix(frame: pd.DataFrame, prefix: str, exclude: set[str]) -> pd.DataFrame:
    renamed = {
        col: f"{prefix}{col}"
        for col in frame.columns
        if col not in exclude
    }
    return frame.rename(columns=renamed)


def enrich_public_data(
    input_csv: Path,
    population_html: Path,
    working_age_html: Path,
    output_csv: Path,
    metadata_out: Path,
) -> None:
    baza = pd.read_csv(input_csv)
    for required in ["bg_admin1_name", "bg_admin2_name"]:
        if required not in baza.columns:
            raise ValueError(f"{required} not found. Run build_bulgaria_geo_features.py first.")

    baza["nsi_district_norm"] = baza["bg_admin1_name"].map(normalize_name)
    baza["nsi_municipality_norm"] = baza["bg_admin2_name"].map(normalize_name)

    pop_district, pop_municipality = load_population_table(population_html)
    age_district, age_municipality = load_working_age_table(working_age_html)

    district_public = pop_district.merge(
        age_district.drop(columns=["area"], errors="ignore"),
        on=["district_norm", "area_norm"],
        how="left",
        suffixes=("", "_age"),
    )
    municipality_public = pop_municipality.merge(
        age_municipality.drop(columns=["area"], errors="ignore"),
        on=["district_norm", "area_norm"],
        how="left",
        suffixes=("", "_age"),
    )

    district_public = add_prefix(district_public, "district_", {"district_norm", "area_norm"})
    municipality_public = add_prefix(municipality_public, "municipality_", {"district_norm", "area_norm"})

    enriched = baza.merge(
        district_public,
        left_on="nsi_district_norm",
        right_on="area_norm",
        how="left",
    ).drop(columns=["area_norm"], errors="ignore")

    enriched = enriched.merge(
        municipality_public,
        left_on=["nsi_district_norm", "nsi_municipality_norm"],
        right_on=["district_norm", "area_norm"],
        how="left",
    ).drop(columns=["district_norm", "area_norm"], errors="ignore")

    enriched["nsi_district_found"] = enriched["district_nsi_population_total"].notna().astype(int)
    enriched["nsi_municipality_found"] = enriched["municipality_nsi_population_total"].notna().astype(int)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_csv, index=False, encoding="utf-8-sig")

    public_cols = [col for col in enriched.columns if col.startswith(("district_nsi_", "municipality_nsi_", "nsi_"))]
    metadata = {
        "source": "National Statistical Institute of Bulgaria public demographic tables",
        "population_source_url": NSI_POPULATION_URL,
        "working_age_source_url": NSI_WORKING_AGE_URL,
        "reference_date": "2023-12-31",
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows": int(len(enriched)),
        "district_matched_rows": int(enriched["nsi_district_found"].sum()),
        "district_match_rate": float(enriched["nsi_district_found"].mean()),
        "municipality_matched_rows": int(enriched["nsi_municipality_found"].sum()),
        "municipality_match_rate": float(enriched["nsi_municipality_found"].mean()),
        "district_public_rows": int(len(district_public)),
        "municipality_public_rows": int(len(municipality_public)),
        "added_columns": public_cols,
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[done] rows={len(enriched)} "
        f"district_match={metadata['district_match_rate']:.2%} "
        f"municipality_match={metadata['municipality_match_rate']:.2%} "
        f"output={output_csv}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=processed_data_path("baza_telecom_v2_bulgaria_geo.csv"))
    parser.add_argument("--population-html", type=Path, default=raw_data_path("nsi_population_municipality_2023.html"))
    parser.add_argument("--working-age-html", type=Path, default=raw_data_path("nsi_working_age_municipality_2023.html"))
    parser.add_argument("--output", type=Path, default=processed_data_path("baza_telecom_v2_bulgaria_public.csv"))
    parser.add_argument("--metadata-out", type=Path, default=processed_data_path("baza_telecom_v2_bulgaria_public_metadata.json"))
    args = parser.parse_args()

    args.input = resolve_input_path(args.input, processed_data_path("baza_telecom_v2_bulgaria_geo.csv"))
    args.population_html = resolve_input_path(args.population_html, raw_data_path("nsi_population_municipality_2023.html"))
    args.working_age_html = resolve_input_path(args.working_age_html, raw_data_path("nsi_working_age_municipality_2023.html"))
    args.output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    args.metadata_out = args.metadata_out if args.metadata_out.is_absolute() else REPO_ROOT / args.metadata_out

    for path in [args.input, args.population_html, args.working_age_html]:
        if not path.is_file():
            raise SystemExit(f"Input not found: {path}")

    enrich_public_data(args.input, args.population_html, args.working_age_html, args.output, args.metadata_out)


if __name__ == "__main__":
    main()
