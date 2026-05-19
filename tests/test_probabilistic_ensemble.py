import unittest

import numpy as np

from src.models.ensemble import AveragingProbabilisticEnsemble


class FixedProbaEstimator:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def predict_proba(self, X):
        scores = self.scores[: len(X)]
        return np.column_stack([1.0 - scores, scores])


class AveragingProbabilisticEnsembleTest(unittest.TestCase):
    def test_predict_proba_averages_range_normalized_scores(self):
        ensemble = AveragingProbabilisticEnsemble(
            [
                ("a", FixedProbaEstimator([0.2, 0.4, 0.6])),
                ("b", FixedProbaEstimator([0.1, 0.2, 0.3])),
            ],
            score_ranges={
                "a": (0.2, 0.6),
                "b": (0.1, 0.3),
            },
        )

        proba = ensemble.predict_proba([0, 1, 2])

        np.testing.assert_allclose(proba[:, 1], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(proba[:, 0], [1.0, 0.5, 0.0])

    def test_predict_uses_half_probability_threshold(self):
        ensemble = AveragingProbabilisticEnsemble([
            ("a", FixedProbaEstimator([0.2, 0.7])),
        ])

        np.testing.assert_array_equal(ensemble.predict([0, 1]), [0, 1])


if __name__ == "__main__":
    unittest.main()
