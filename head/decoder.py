import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Tuple


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out += residual

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256]):
        super(Decoder, self).__init__()

        self.res_1 = ResBlock(channels[0] * 2, 1)
        self.res_2 = ResBlock(channels[1] * 2, channels[0])
        self.res_3 = ResBlock(channels[2] * 2, channels[1])
        self.res_4 = ResBlock(channels[3], channels[2])
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, feature_fused):
        z1, z2, z3, z4 = feature_fused

        out = self.res_4(z4)
        out = self.up(out)
        out = torch.cat((out, z3), dim=1)

        out = self.res_3(out)
        out = self.up(out)
        out = torch.cat((out, z2), dim=1)

        out = self.res_2(out)
        out = self.up(out)
        out = torch.cat((out, z1), dim=1)

        out = self.res_1(out)
        
        return out
