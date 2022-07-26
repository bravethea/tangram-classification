import random
import unittest

import torch

from ..model import PoseTaggingModel


def get_batch(bsz=4):
    data_tensor = torch.tensor([[[1, 1]], [[2, 2]], [[3, 3]]], dtype=torch.float)
    return {
        "pose": {
            "data": data_tensor.expand(bsz, *data_tensor.shape),
        },
        "mask": torch.ones([bsz, 3], dtype=torch.float),
        "lh": torch.stack([torch.tensor([4, 2, 1], dtype=torch.long)] * bsz),
        "rh": torch.stack([torch.tensor([1, 1, 3], dtype=torch.long)] * bsz),
    }


class ModelOverfitTestCase(unittest.TestCase):
    def test_model_should_overfit(self):
        torch.manual_seed(42)
        random.seed(42)

        batch = get_batch(bsz=1)

        model = PoseTaggingModel(
            class_weights=[1.01, 1, 1, 1, 1],
            hidden_dim=10,
            pose_dims=(1, 2),
        )
        optimizer = model.configure_optimizers()

        model.train()
        torch.set_grad_enabled(True)

        # Training loop
        losses = []
        for _ in range(200):
            loss = model.training_step(batch)
            loss_float = float(loss.detach())
            losses.append(loss_float)

            optimizer.zero_grad()  # clear gradients
            loss.backward()  # backward
            optimizer.step()  # update parameters

        print("losses", losses)

        pose_data = batch["pose"]["data"][0].unsqueeze(0)
        prob = model(pose_data)

        sign_argmax = torch.argmax(prob["lh"], dim=-1)
        print("lh_argmax", sign_argmax)
        self.assertTrue(torch.all(torch.eq(sign_argmax, batch["lh"][0])))

        sentence_argmax = torch.argmax(prob["rh"], dim=-1)
        print("rh_argmax", sentence_argmax)
        self.assertTrue(torch.all(torch.eq(sentence_argmax, batch["rh"][0])))


if __name__ == '__main__':
    unittest.main()
