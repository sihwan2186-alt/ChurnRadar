from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_CHURN_THRESHOLD = 0.5


def coerce_threshold(value: Any, default: float = DEFAULT_CHURN_THRESHOLD) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return default

    if 0.0 <= threshold <= 1.0:
        return threshold
    return default


def threshold_from_payload(
    payload: dict[str, Any],
    default: float = DEFAULT_CHURN_THRESHOLD,
) -> float:
    if "selected_threshold" in payload:
        return coerce_threshold(payload["selected_threshold"], default)

    best = payload.get("best")
    if isinstance(best, dict) and "threshold" in best:
        return coerce_threshold(best["threshold"], default)

    best_validation = payload.get("best_validation")
    if isinstance(best_validation, dict) and "threshold" in best_validation:
        return coerce_threshold(best_validation["threshold"], default)

    return default


def resolve_churn_threshold(
    threshold_path: Path,
    env_var: str = "CHURN_THRESHOLD",
    default: float = DEFAULT_CHURN_THRESHOLD,
) -> float:
    env_value = os.getenv(env_var)
    if env_value not in (None, ""):
        return coerce_threshold(env_value, default)

    if not threshold_path.is_file():
        return default

    try:
        payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default

    return threshold_from_payload(payload, default)
