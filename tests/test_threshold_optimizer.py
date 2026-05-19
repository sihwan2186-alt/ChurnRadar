import json
import tempfile
import unittest
from pathlib import Path

from api.threshold_policy import resolve_churn_threshold, threshold_from_payload
from src.utils.threshold_optimizer import evaluate_threshold, find_best_threshold


class ThresholdOptimizerTest(unittest.TestCase):
    def test_find_best_threshold_prefers_f1_on_validation_scores(self):
        y_true = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
        scores = [0.82, 0.48, 0.41, 0.39, 0.36, 0.34, 0.28, 0.21, 0.18, 0.09]

        best = find_best_threshold(y_true, scores, thresholds=[0.5, 0.35, 0.3])

        self.assertEqual(best.threshold, 0.3)
        self.assertAlmostEqual(best.recall, 1.0)
        self.assertAlmostEqual(best.f1, 0.8)

    def test_evaluate_threshold_counts_confusion_matrix_terms(self):
        metrics = evaluate_threshold(
            y_true=[1, 1, 0, 0],
            scores=[0.8, 0.4, 0.7, 0.1],
            threshold=0.5,
        )

        self.assertEqual(metrics.tp, 1)
        self.assertEqual(metrics.fp, 1)
        self.assertEqual(metrics.fn, 1)
        self.assertEqual(metrics.tn, 1)

    def test_threshold_payload_supports_selected_threshold(self):
        self.assertEqual(
            threshold_from_payload({"selected_threshold": 0.37}),
            0.37,
        )

    def test_resolve_threshold_reads_json_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "threshold.json"
            path.write_text(json.dumps({"selected_threshold": 0.42}), encoding="utf-8")

            self.assertEqual(resolve_churn_threshold(path), 0.42)


if __name__ == "__main__":
    unittest.main()
