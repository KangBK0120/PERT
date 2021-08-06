from typing import List

import torch
import torch.nn as nn

from .blocks import ConvBlock, ResidualBlock


class BackBone(nn.Module):
    def __init__(self):
        super(BackBone, self).__init__()

        self.convblock1 = ConvBlock(
            in_channels=6, out_channels=32, kernel_size=4, stride=2, padding=1
        )
        self.convblock2 = ConvBlock(
            in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1
        )

        self.residual_block1 = ResidualBlock(in_channels=32, out_channels=64, downsample=False)
        self.residual_block2 = ResidualBlock(in_channels=64, out_channels=128, downsample=True)
        self.residual_block3 = ResidualBlock(in_channels=128, out_channels=128, downsample=True)
        self.residual_block4 = ResidualBlock(in_channels=128, out_channels=256, downsample=False)
        self.residual_block5 = ResidualBlock(in_channels=256, out_channels=256, downsample=True)

    def forward(self, concat_imgs: torch.Tensor) -> List[torch.Tensor]:
        out = self.convblock1(concat_imgs)
        conv1_output = out.clone()
        out = self.convblock2(out)

        out = self.residual_block1(out)
        res1_output = out.clone()
        out = self.residual_block2(out)
        res2_output = out.clone()

        out = self.residual_block3(out)

        out = self.residual_block4(out)
        res4_output = out.clone()
        out = self.residual_block5(out)

        return [conv1_output, res1_output, res2_output, res4_output, out]
