import unittest
from datetime import datetime, timedelta, timezone

from api.alert_fatigue import classify_risk_level, evaluate_alert_fatigue


class AlertFatigueTest(unittest.TestCase):
    def test_high_risk_duplicate_is_suppressed_within_three_days(self):
        now = datetime(2026, 5, 14, 9, 0, tzinfo=timezone.utc)

        decision = evaluate_alert_fatigue(
            risk_level="High",
            churn_probability=0.72,
            last_alert_time=(now - timedelta(days=1)).isoformat(),
            now=now,
        )

        self.assertFalse(decision.alert_required)
        self.assertEqual(decision.alert_channel, "None")
        self.assertIn("최근 3일", decision.suppress_reason)
        self.assertTrue(decision.log_required)

    def test_critical_risk_allows_renotify_after_24_hours_without_response(self):
        now = datetime(2026, 5, 14, 9, 0, tzinfo=timezone.utc)

        decision = evaluate_alert_fatigue(
            risk_level="Critical",
            churn_probability=0.91,
            last_alert_time=(now - timedelta(hours=25)).isoformat(),
            response_status="미대응",
            now=now,
        )

        self.assertTrue(decision.alert_required)
        self.assertEqual(decision.alert_channel, "Slack")
        self.assertIsNone(decision.suppress_reason)

    def test_probability_spike_overrides_recent_duplicate_window(self):
        now = datetime(2026, 5, 14, 9, 0, tzinfo=timezone.utc)

        decision = evaluate_alert_fatigue(
            risk_level="Critical",
            churn_probability=0.81,
            previous_churn_probability=0.45,
            last_alert_time=(now - timedelta(hours=2)).isoformat(),
            now=now,
        )

        self.assertTrue(decision.alert_required)
        self.assertEqual(decision.alert_channel, "Slack")
        self.assertIsNone(decision.suppress_reason)

    def test_medium_risk_is_logged_without_realtime_alert(self):
        decision = evaluate_alert_fatigue(
            risk_level="Medium",
            churn_probability=0.58,
        )

        self.assertFalse(decision.alert_required)
        self.assertEqual(decision.alert_channel, "None")
        self.assertIn("Google Sheets", decision.suppress_reason)

    def test_risk_level_thresholds_match_alert_policy(self):
        self.assertEqual(classify_risk_level(0.81), "Critical")
        self.assertEqual(classify_risk_level(0.72), "High")
        self.assertEqual(classify_risk_level(0.51), "Medium")
        self.assertEqual(classify_risk_level(0.49), "Low")


if __name__ == "__main__":
    unittest.main()
