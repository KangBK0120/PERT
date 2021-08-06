import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding

from .blocks import ConvBlock
from .pspmodule import PSPModule


class TLN(nn.Module):
    def __init__(self):
        super(TLN, self).__init__()

        self.pspmodule = PSPModule(256, 512)
        self.conv1 = ConvBlock(
            in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = ConvBlock(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = ConvBlock(
            in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.conv4 = ConvBlock(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvBlock(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(
        self, feat: torch.Tensor, res2_feat: torch.Tensor, res4_feat: torch.Tensor
    ) -> torch.Tensor:
        out = self.pspmodule(feat)
        out = self.conv1(out)
        out = F.interpolate(out, scale_factor=2, mode="bilinear")
        out = self.conv2(out)

        out = torch.cat([out, res4_feat], dim=1)
        out = F.interpolate(out, scale_factor=2, mode="bilinear")
        out = self.conv3(out)

        out = torch.cat([out, res2_feat], dim=1)
        out = F.interpolate(out, scale_factor=2, mode="bilinear")
        out = self.conv4(out)
        out = self.conv5(out)

        out = F.interpolate(out, scale_factor=4, mode="bilinear")
        mask = F.sigmoid(out)

        return mask
