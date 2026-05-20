import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.make_baza_ts import simulate_baza_sequence
from src.preprocessing.cleaner import clean_data


POLARS_AVAILABLE = importlib.util.find_spec("polars") is not None


class FeatureImprovementTest(unittest.TestCase):
    def test_baza_static_engineered_features_are_created(self):
        raw = pd.DataFrame(
            {
                "Total_SUBs": [4],
                "AvgMobileRevenue": [70.0],
                "AvgFIXRevenue": [30.0],
                "TotalRevenue": [100.0],
                "ARPU": [25.0],
                "Active_subscribers": [2],
                "Not_Active_subscribers": [1.0],
                "Suspended_subscribers": [1.0],
                "CRM_PID_Value_Segment": ["VIP"],
                "EffectiveSegment": ["Business"],
                "CHURN": ["Yes"],
            }
        )

        X, y = clean_data(raw)

        self.assertEqual(y.iloc[0], 1)
        self.assertAlmostEqual(X.loc[0, "Suspended_Ratio"], 0.25)
        self.assertAlmostEqual(X.loc[0, "Revenue_per_Active_Sub"], 50.0)
        self.assertAlmostEqual(X.loc[0, "Inactive_x_Revenue"], 25.0)
        self.assertAlmostEqual(X.loc[0, "Revenue_Balance"], 30.0 / (70.0 + 1e-5))

    def test_baza_churn_sequence_has_directional_trend(self):
        rng = np.random.default_rng(7)

        energy, momentum, accel = simulate_baza_sequence(
            base_energy=100.0,
            base_momentum=0.8,
            base_accel=0.2,
            is_churn=True,
            rng=rng,
            time_steps=30,
            noise_std=0.0,
        )

        self.assertLess(energy[-1], energy[0])
        self.assertLess(momentum[-1], momentum[0])
        self.assertGreater(accel[-1], accel[0])

    def test_timeseries_dataset_accepts_behavior_channels(self):
        if not POLARS_AVAILABLE:
            self.skipTest("polars is not installed")
        import polars as pl
        from src.data.ts_dataset import ChurnTimeSeriesDataset

        rows = []
        for entity_id, label in [("A", 1), ("B", 0)]:
            for t in range(2):
                rows.append(
                    {
                        "Entity_ID": entity_id,
                        "Event_Time": t,
                        "Activity_Energy": 10.0 + t,
                        "Momentum": 1.0,
                        "Acceleration": 0.0,
                        "Skip_Rate": 0.1 * t,
                        "Completion_Rate": 0.8,
                        "Diversity_Score": 0.5,
                        "is_churn": label,
                    }
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ts.parquet"
            pl.DataFrame(rows).write_parquet(path)
            dataset = ChurnTimeSeriesDataset(path, max_seq_len=3, target_col="is_churn")

        self.assertEqual(dataset.X.shape, (2, 3, 6))
        self.assertEqual(dataset.feature_cols[-3:], ["Skip_Rate", "Completion_Rate", "Diversity_Score"])

    def test_kkbox_mapper_outputs_behavior_channels(self):
        if not POLARS_AVAILABLE:
            self.skipTest("polars is not installed")
        import polars as pl
        from src.preprocessing.cross_domain_mapper import map_kkbox_to_energy

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "logs.csv"
            out_path = tmp / "kkbox.parquet"
            pd.DataFrame(
                {
                    "msno": ["u1", "u1"],
                    "date": [20170330, 20170331],
                    "num_25": [1, 2],
                    "num_50": [0, 0],
                    "num_75": [0, 1],
                    "num_985": [0, 0],
                    "num_100": [3, 2],
                    "num_unq": [4, 5],
                    "total_secs": [100.0, 80.0],
                }
            ).to_csv(csv_path, index=False)

            map_kkbox_to_energy(csv_path, out_path)
            mapped = pl.read_parquet(out_path)

        self.assertIn("Skip_Rate", mapped.columns)
        self.assertIn("Completion_Rate", mapped.columns)
        self.assertIn("Diversity_Score", mapped.columns)
        self.assertEqual(mapped.height, 2)


if __name__ == "__main__":
    unittest.main()
