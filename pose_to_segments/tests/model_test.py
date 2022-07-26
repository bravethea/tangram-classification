import unittest
from unittest.mock import MagicMock

import torch

from ..model import PoseTaggingModel


class ModelTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = [1.01, 1, 1, 1, 1]
        self.pose_dim = (2, 2)
        self.seq_length = 5
        self.hidden_dim = 4

    def model_setup(self):
        model = PoseTaggingModel(
            class_weights=self.class_weights,
            hidden_dim=self.hidden_dim,
            pose_dims=self.pose_dim,
            encoder_depth=2
        )
        model.log = MagicMock(return_value=True)
        return model

    def get_batch(self):
        return {
            "pose": {
                "data": torch.ones([2, self.seq_length, *self.pose_dim], dtype=torch.float),
            },
            "mask": torch.ones([2, self.seq_length], dtype=torch.float),
            "lh": torch.zeros((2, self.seq_length), dtype=torch.long),
            "rh": torch.zeros((2, self.seq_length), dtype=torch.long)
        }

    def test_forward_yields_bio_probs(self):
        model = self.model_setup()
        batch = self.get_batch()
        log_probs = model.forward(batch["pose"]["data"])

        # shape check
        self.assertEqual(log_probs["lh"].shape, (len(batch["lh"]), self.seq_length, 5))
        self.assertEqual(log_probs["rh"].shape, (len(batch["rh"]), self.seq_length, 5))

        # nan / inf check
        self.assertTrue(torch.all(torch.isfinite(log_probs["lh"])))
        self.assertTrue(torch.all(torch.isfinite(log_probs["rh"])))

        # softmax probs check
        sum_lh = torch.exp(log_probs["lh"]).sum(-1)
        self.assertTrue(torch.allclose(sum_lh, torch.ones_like(sum_lh)))
        sum_rh = torch.exp(log_probs["rh"]).sum(-1)
        self.assertTrue(torch.allclose(sum_rh, torch.ones_like(sum_rh)))

    def test_training_step_expected_loss_finite(self):
        model = self.model_setup()
        batch = self.get_batch()

        loss = model.training_step(batch)
        self.assertNotEqual(float(loss), 0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
