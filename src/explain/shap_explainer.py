"""SHAP 기반 피처 중요도 분석 및 시각화."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def get_shap_values(model: Pipeline, X: pd.DataFrame) -> tuple[shap.Explanation, np.ndarray]:
    """Pipeline에서 전처리 후 SHAP 값 계산."""
    prep = model[:-1]   # 마지막 스텝(모델) 제외한 전처리기
    clf = model[-1]     # 마지막 스텝(모델)
    X_transformed = prep.transform(X)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_transformed)
    return shap_values, X_transformed


def get_feature_names(model: Pipeline, numeric_features: list, cat_features: list) -> list[str]:
    """ColumnTransformer 출력 순서에 맞는 피처명 반환."""
    return numeric_features + cat_features


def plot_summary(
    model: Pipeline,
    X: pd.DataFrame,
    numeric_features: list[str],
    cat_features: list[str],
    save_path: Path | None = None,
    max_display: int = 15,
) -> None:
    """SHAP Summary Plot (beeswarm) 출력."""
    shap_values, _ = get_shap_values(model, X)
    feature_names = get_feature_names(model, numeric_features, cat_features)

    plt.figure(figsize=(10, 6))
    # 이진 분류: class 1 (Churn) 기준
    sv = shap_values[..., 1] if shap_values.values.ndim == 3 else shap_values
    shap.summary_plot(sv, feature_names=feature_names, max_display=max_display, show=False)
    plt.title("SHAP Feature Importance (Churn=1)", fontsize=13)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"저장: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_bar(
    model: Pipeline,
    X: pd.DataFrame,
    numeric_features: list[str],
    cat_features: list[str],
    save_path: Path | None = None,
    max_display: int = 15,
) -> pd.Series:
    """SHAP 평균 절대값 기준 피처 중요도 bar plot 및 반환."""
    shap_values, _ = get_shap_values(model, X)
    feature_names = get_feature_names(model, numeric_features, cat_features)

    sv = shap_values[..., 1] if shap_values.values.ndim == 3 else shap_values
    mean_abs = pd.Series(
        np.abs(sv.values).mean(axis=0),
        index=feature_names,
    ).sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    mean_abs.head(max_display).plot(kind="barh").invert_yaxis()
    plt.xlabel("Mean |SHAP value|")
    plt.title("SHAP Feature Importance (Mean |SHAP|)", fontsize=13)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"저장: {save_path}")
    else:
        plt.show()
    plt.close()

    return mean_abs
