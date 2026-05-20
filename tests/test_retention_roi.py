import unittest

from api.retention_roi import calculate_retention_roi


class RetentionROITest(unittest.TestCase):
    def test_successful_retention_calculates_saved_revenue_and_roi(self):
        result = calculate_retention_roi(
            customer_id="B2B-1023",
            churn_probability=0.82,
            risk_level="Critical",
            alert_sent=True,
            expected_revenue_loss=1000.0,
            action_type="discount",
            discount_cost=120.0,
            consulting_cost=30.0,
            response_status="completed",
            actual_churn=False,
        )

        self.assertTrue(result.retention_success)
        self.assertEqual(result.saved_revenue, 1000.0)
        self.assertEqual(result.retention_cost, 150.0)
        self.assertEqual(result.net_benefit, 850.0)
        self.assertAlmostEqual(result.roi, 850.0 / 150.0)

    def test_failed_retention_keeps_cost_but_no_saved_revenue(self):
        result = calculate_retention_roi(
            customer_id="B2B-2081",
            churn_probability=0.74,
            risk_level="High",
            alert_sent=True,
            expected_revenue_loss=500.0,
            action_cost=80.0,
            response_status="completed",
            actual_churn=True,
        )

        self.assertFalse(result.retention_success)
        self.assertEqual(result.saved_revenue, 0.0)
        self.assertEqual(result.retention_cost, 80.0)
        self.assertEqual(result.net_benefit, -80.0)
        self.assertEqual(result.roi, -1.0)

    def test_zero_cost_returns_none_roi_to_avoid_division_by_zero(self):
        result = calculate_retention_roi(
            customer_id="B2B-3302",
            churn_probability=0.91,
            risk_level="Critical",
            alert_sent=True,
            expected_revenue_loss=1200.0,
            retention_success=True,
        )

        self.assertTrue(result.retention_success)
        self.assertEqual(result.saved_revenue, 1200.0)
        self.assertEqual(result.retention_cost, 0.0)
        self.assertIsNone(result.roi)


if __name__ == "__main__":
    unittest.main()
