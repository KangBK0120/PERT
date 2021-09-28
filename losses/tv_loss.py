import torch
import torch.nn as nn
from kornia.losses import total_variation


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C, H, W = x.size(1), x.size(2), x.size(3)
        return total_variation(x).mean() / (C * H * W)
