import torch
import torch.nn as nn
import torch.nn.functional as F


class LateFusionDeep(nn.Module):
    def __init__(
        self,
        img_dim: int,
        depth_dim: int,
        text_dim: int,
        num_classes: int,
        dropout: float = 0.3,
    ):
        super().__init__()

        input_dim = img_dim + depth_dim + text_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, img, depth, text):
        fused = torch.cat([img, depth, text], dim=1)
        fused = F.normalize(fused, dim=1)
        return self.net(fused)
