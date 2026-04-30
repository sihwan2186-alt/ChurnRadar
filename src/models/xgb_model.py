from xgboost import XGBClassifier

# 클래스 불균형 비율: No(7904) / Yes(549) ≈ 14.4
DEFAULT_SCALE_POS_WEIGHT = 14.4


def build_xgb(scale_pos_weight: float = DEFAULT_SCALE_POS_WEIGHT, **kwargs) -> XGBClassifier:
    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }
    params.update(kwargs)
    return XGBClassifier(**params)
