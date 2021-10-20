import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        have_relu: bool = True,
    ):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        if have_relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if self.relu:
            return self.relu(self.conv(feat))
        else:
            return self.conv(feat)


class DeConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DeConvBlock, self).__init__()

        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1
        )
        self.relu = nn.ReLU()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.relu(self.deconv(feat))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False):
        super(ResidualBlock, self).__init__()
        conv1_stride = 1 if not downsample else 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=conv1_stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=2 if downsample else 1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        residual = feat
        out = self.block(feat)
        if self.downsample:
            residual = self.downsample(feat)
        out += residual
        out = F.relu(out)
        return out
