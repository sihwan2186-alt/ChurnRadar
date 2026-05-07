from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = REPO_ROOT / "raw" / "raw"
LEGACY_RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
INTERIM_DATA_DIR = REPO_ROOT / "data" / "interim"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = REPO_ROOT / "plots"


def first_existing_path(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def raw_data_path(filename: str) -> Path:
    preferred = RAW_DATA_DIR / filename
    legacy = LEGACY_RAW_DATA_DIR / filename
    return first_existing_path(preferred, legacy) or preferred


def processed_data_path(filename: str) -> Path:
    return PROCESSED_DATA_DIR / filename


def interim_data_path(filename: str) -> Path:
    return INTERIM_DATA_DIR / filename


def model_path(filename: str) -> Path:
    return MODELS_DIR / filename


def result_path(filename: str) -> Path:
    return RESULTS_DIR / filename


def plot_path(filename: str) -> Path:
    return PLOTS_DIR / filename


def resolve_input_path(path: str | Path, *fallbacks: Path) -> Path:
    requested = Path(path)
    if not requested.is_absolute():
        requested = REPO_ROOT / requested

    for candidate in (requested, *fallbacks):
        if candidate.exists():
            return candidate

    return requested
