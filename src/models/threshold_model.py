from __future__ import annotations

from typing import Any

import numpy as np


class ThresholdClassifier:
    """Wrap a probabilistic classifier with an F1-tuned decision threshold."""

    def __init__(self, estimator: Any, threshold: float = 0.5) -> None:
        self.estimator = estimator
        self.threshold = threshold

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def decision_function(self, X):
        if hasattr(self.estimator, "decision_function"):
            return self.estimator.decision_function(X)
        return self.predict_proba(X)[:, 1]

    def predict(self, X):
        scores = self.predict_proba(X)[:, 1]
        return np.asarray(scores >= self.threshold, dtype=int)
