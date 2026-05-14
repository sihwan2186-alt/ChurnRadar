import importlib.util
import unittest


@unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
class ChurnTCNTest(unittest.TestCase):
    def test_forward_returns_one_logit_per_customer(self):
        import torch

        from src.models.tcn_model import ChurnTCN

        model = ChurnTCN(input_size=3, channels=(8, 16), kernel_size=3, dropout=0.0)
        x = torch.randn(4, 30, 3)
        lengths = torch.tensor([30, 24, 12, 7])

        logits = model(x, lengths=lengths)

        self.assertEqual(tuple(logits.shape), (4, 1))

    def test_padding_mask_path_matches_expected_shape(self):
        import torch

        from src.models.tcn_model import ChurnTCN

        model = ChurnTCN(input_size=3, channels=(8,), kernel_size=3, dropout=0.0)
        x = torch.randn(2, 30, 3)
        padding_mask = torch.zeros(2, 30, dtype=torch.bool)
        padding_mask[1, 10:] = True

        logits = model(x, padding_mask=padding_mask)

        self.assertEqual(tuple(logits.shape), (2, 1))


if __name__ == "__main__":
    unittest.main()
