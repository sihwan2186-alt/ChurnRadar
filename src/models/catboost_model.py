from catboost import CatBoostClassifier

# baza_telecom_v2.csv 기준 범주형 피처
DEFAULT_CAT_FEATURES = ["CRM_PID_Value_Segment", "EffectiveSegment", "KA_name"]


def build_catboost(cat_features: list = None, **kwargs) -> CatBoostClassifier:
    """CatBoost 분류기 생성. 범주형 피처를 자동으로 처리합니다."""
    params = {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "eval_metric": "F1",
        "auto_class_weights": "Balanced",
        "random_seed": 42,
        "verbose": 0,
    }
    params.update(kwargs)
    return CatBoostClassifier(
        cat_features=cat_features if cat_features is not None else DEFAULT_CAT_FEATURES,
        **params,
    )
