from sklearn.linear_model import LogisticRegression


def build_lr(**kwargs) -> LogisticRegression:
    params = {
        "max_iter": 2000,
        "class_weight": "balanced",
        "random_state": 42,
    }
    params.update(kwargs)
    return LogisticRegression(**params)
