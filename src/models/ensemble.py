from __future__ import annotations

import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline


def build_voting_ensemble(named_pipelines: dict) -> VotingClassifier:
    """학습된 Pipeline들로 Soft Voting 앙상블 구성.

    Args:
        named_pipelines: {"lr": Pipeline, "rf": Pipeline, ...}
    Returns:
        VotingClassifier (soft voting)
    """
    estimators = list(named_pipelines.items())
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


class AveragingProbabilisticEnsemble:
    """Average already-fitted probabilistic estimators.

    This is useful when each member was trained on a different source-domain
    mixture and therefore cannot be refit together by sklearn VotingClassifier.
    """

    def __init__(
        self,
        estimators: list[tuple[str, object]],
        weights: list[float] | None = None,
        score_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if not estimators:
            raise ValueError("At least one estimator is required.")
        self.estimators = estimators
        self.weights = np.asarray(weights if weights is not None else np.ones(len(estimators)), dtype=float)
        if len(self.weights) != len(self.estimators):
            raise ValueError("weights must match the number of estimators.")
        if np.allclose(self.weights.sum(), 0):
            raise ValueError("weights must not sum to zero.")
        self.weights = self.weights / self.weights.sum()
        self.score_ranges = score_ranges or {}

    def _positive_score(self, name: str, estimator: object, X) -> np.ndarray:
        if not hasattr(estimator, "predict_proba"):
            raise TypeError(f"Estimator {name} does not provide predict_proba.")
        proba = estimator.predict_proba(X)
        if proba.ndim != 2 or proba.shape[1] < 2:
            scores = np.asarray(proba).ravel().astype(float)
        else:
            scores = np.asarray(proba[:, 1], dtype=float)

        score_range = self.score_ranges.get(name)
        if score_range is None:
            return np.clip(scores, 0.0, 1.0)
        lo, hi = score_range
        if hi - lo < 1e-12:
            return np.zeros_like(scores, dtype=float)
        return np.clip((scores - lo) / (hi - lo), 0.0, 1.0)

    def predict_proba(self, X) -> np.ndarray:
        member_scores = [
            self._positive_score(name, estimator, X)
            for name, estimator in self.estimators
        ]
        positive = np.average(np.vstack(member_scores), axis=0, weights=self.weights)
        positive = np.clip(positive, 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
