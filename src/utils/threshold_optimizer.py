from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def evaluate_threshold(
    y_true: Sequence[int | float],
    scores: Sequence[float],
    threshold: float,
) -> ThresholdMetrics:
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")

    tp = fp = fn = tn = 0
    for truth, score in zip(y_true, scores):
        actual = int(truth) == 1
        predicted = float(score) >= threshold
        if actual and predicted:
            tp += 1
        elif not actual and predicted:
            fp += 1
        elif actual and not predicted:
            fn += 1
        else:
            tn += 1

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return ThresholdMetrics(
        threshold=float(threshold),
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def evaluate_thresholds(
    y_true: Sequence[int | float],
    scores: Sequence[float],
    thresholds: Iterable[float],
) -> list[ThresholdMetrics]:
    return [evaluate_threshold(y_true, scores, threshold) for threshold in thresholds]


def find_best_threshold(
    y_true: Sequence[int | float],
    scores: Sequence[float],
    thresholds: Iterable[float],
    min_recall: float | None = None,
) -> ThresholdMetrics:
    candidates = evaluate_thresholds(y_true, scores, thresholds)
    if min_recall is not None:
        recall_filtered = [item for item in candidates if item.recall >= min_recall]
        if recall_filtered:
            candidates = recall_filtered
    if not candidates:
        return evaluate_threshold(y_true, scores, 0.5)

    # Optimize F1 first. If tied, prefer higher recall for churn operations.
    return max(candidates, key=lambda item: (item.f1, item.recall, -item.threshold))
