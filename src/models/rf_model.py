from sklearn.ensemble import RandomForestClassifier


def build_rf(**kwargs) -> RandomForestClassifier:
    params = {
        "n_estimators": 300,
        "max_depth": 10,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    params.update(kwargs)
    return RandomForestClassifier(**params)
