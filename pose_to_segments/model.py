from typing import List

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from pose_to_segments.data import CLASSES


class PoseTaggingModel(pl.LightningModule):
    def __init__(
            self,
            class_weights: List[float],
            pose_dims: (int, int) = (137, 2),
            hidden_dim: int = 128,
            encoder_depth=2):
        super().__init__()

        self.pose_dims = pose_dims
        pose_dim = int(np.prod(pose_dims))

        self.pose_projection = nn.Linear(pose_dim, hidden_dim)

        assert hidden_dim / 2 == hidden_dim // 2, "Hidden dimensions must be even, not odd"

        # Encoder
        self.encoder = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=encoder_depth,
                               batch_first=True, bidirectional=True)

        # tag sequence for sign bio / sentence bio
        self.lh_head = nn.Linear(hidden_dim, len(CLASSES))
        self.rh_head = nn.Linear(hidden_dim, len(CLASSES))

        loss_weight = torch.tensor(class_weights, dtype=torch.float)
        # loss_weight /= loss_weight.sum()
        self.loss_function = nn.NLLLoss(reduction='none', weight=loss_weight)

    def forward(self, pose_data: torch.Tensor):
        batch_size, seq_length, _, _ = pose_data.shape
        flat_pose_data = pose_data.reshape(batch_size, seq_length, -1)

        pose_projection = self.pose_projection(flat_pose_data)
        pose_encoding, _ = self.encoder(pose_projection)

        lh_logits = self.lh_head(pose_encoding)
        rh_logits = self.rh_head(pose_encoding)

        return {
            "lh": F.log_softmax(lh_logits, dim=-1),
            "rh": F.log_softmax(rh_logits, dim=-1)
        }

    def training_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, name="train")

    def validation_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, name="validation")

    def step(self, batch, *unused_args, name: str):
        pose_data = batch["pose"]["data"]
        batch_size = len(pose_data)

        log_probs = self.forward(pose_data)

        loss_mask = batch["mask"].reshape(-1)

        lh_losses = self.loss_function(log_probs["lh"].reshape(-1, len(CLASSES)), batch["lh"].reshape(-1))
        lh_loss = (lh_losses * loss_mask).mean()
        rh_losses = self.loss_function(log_probs["rh"].reshape(-1, len(CLASSES)), batch["rh"].reshape(-1))
        rh_loss = (rh_losses * loss_mask).mean()
        loss = lh_loss + rh_loss

        self.log(name + "_lh_loss", lh_loss, batch_size=batch_size)
        self.log(name + "_rh_loss", rh_loss, batch_size=batch_size)
        self.log(name + "_loss", loss, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
