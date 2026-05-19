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


def build_balanced_random_forest(**kwargs):
    from imblearn.ensemble import BalancedRandomForestClassifier

    params = {
        "n_estimators": 300,
        "max_depth": 10,
        "sampling_strategy": "all",
        "replacement": True,
        "bootstrap": False,
        "random_state": 42,
        "n_jobs": -1,
    }
    params.update(kwargs)
    return BalancedRandomForestClassifier(**params)


def build_easy_ensemble(**kwargs):
    from imblearn.ensemble import EasyEnsembleClassifier

    params = {
        "n_estimators": 10,
        "random_state": 42,
        "n_jobs": -1,
    }
    params.update(kwargs)
    return EasyEnsembleClassifier(**params)
