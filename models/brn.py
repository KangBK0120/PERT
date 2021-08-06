from typing import List

import torch
import torch.nn as nn

from .blocks import ConvBlock, DeConvBlock, ResidualBlock


class BRN(nn.Module):
    def __init__(self):
        super(BRN, self).__init__()

        self.res1 = ResidualBlock(in_channels=256, out_channels=256, downsample=False)
        self.deconv1 = DeConvBlock(in_channels=256, out_channels=256)
        self.conv1 = ConvBlock(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.deconv2 = DeConvBlock(in_channels=512, out_channels=512)
        self.conv2 = ConvBlock(
            in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.deconv3 = DeConvBlock(in_channels=256, out_channels=256)
        self.conv3 = ConvBlock(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.deconv4 = DeConvBlock(in_channels=128, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.deconv5 = DeConvBlock(in_channels=64, out_channels=64)
        self.conv5 = ConvBlock(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.mrm1 = ConvBlock(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.mrm2 = ConvBlock(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        feat: torch.Tensor,
        conv1_feat: torch.Tensor,
        res1_feat: torch.Tensor,
        res2_feat: torch.Tensor,
        res4_feat: torch.Tensor,
    ) -> List[torch.Tensor]:
        out = self.res1(feat)
        out = self.deconv1(out)
        out = self.conv1(out)

        out = torch.cat([out, res4_feat], dim=1)
        out = self.deconv2(out)
        out = self.conv2(out)

        out = torch.cat([out, res2_feat], dim=1)
        out = self.deconv3(out)
        out = self.conv3(out)

        out = torch.cat([out, res1_feat], dim=1)
        out = self.deconv4(out)
        p1_out = out.clone()
        out = self.conv4(out)

        out = torch.cat([out, conv1_feat], dim=1)
        out = self.deconv5(out)
        p2_out = out.clone()
        out = self.conv5(out)

        p1_out = self.mrm1(p1_out)
        p2_out = self.mrm2(p2_out)

        return [p1_out, p2_out, out]
