from lightgbm import LGBMClassifier


def build_lgbm(**kwargs) -> LGBMClassifier:
    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    params.update(kwargs)
    return LGBMClassifier(**params)
