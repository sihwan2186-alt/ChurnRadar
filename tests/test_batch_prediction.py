import importlib.util
import unittest
from unittest.mock import patch


def sample_customer(customer_id: str):
    from api.schemas import CustomerData

    return CustomerData(
        customer_id=customer_id,
        total_subs=3,
        avg_mobile_revenue=50.0,
        avg_fix_revenue=20.0,
        total_revenue=70.0,
        arpu=23.3,
        active_subscribers=2,
        not_active_subscribers=1.0,
        crm_segment="VIP",
        effective_segment="Business",
    )


@unittest.skipIf(
    importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("pydantic") is None,
    "fastapi/pydantic is not installed",
)
class BatchPredictionTest(unittest.TestCase):
    def test_batch_prediction_reuses_single_customer_policy(self):
        from api.main import predict_batch_endpoint
        from api.schemas import BatchPredictionRequest

        def fake_predict(data):
            probability = 0.83 if data["customer_id"] == "B2B-1" else 0.44
            return {
                "xgb_probability": probability,
                "tcn_probability": None,
                "ts_probability": 0.20,
                "churn_probability": probability,
                "prediction_threshold": 0.5,
                "churn_prediction": probability >= 0.5,
                "risk_level": "Critical" if probability >= 0.8 else "Low",
                "expected_revenue_loss": data["arpu"] if probability >= 0.5 else 0.0,
            }

        request = BatchPredictionRequest(
            batch_id="batch-test",
            customers=[sample_customer("B2B-1"), sample_customer("B2B-2")],
        )

        with patch("api.main.predict_churn", side_effect=fake_predict):
            response = predict_batch_endpoint(request)

        self.assertEqual(response.batch_id, "batch-test")
        self.assertEqual(response.total_customers, 2)
        self.assertEqual(len(response.predictions), 2)
        self.assertEqual(response.predictions[0].risk_level, "Critical")
        self.assertTrue(response.predictions[0].alert_required)
        self.assertFalse(response.predictions[1].alert_required)
        self.assertEqual(response.alert_required_count, 1)


if __name__ == "__main__":
    unittest.main()
