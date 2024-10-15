import torch
from torch import nn, Tensor
from torch.nn import functional as F
from modules.modules import CBAM

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        mid_channels = out_channels
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(mid_channels),
            )
        else:
            self.shortcut = None
        self.att = CBAM(out_channels)

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out += residual

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.att(out)
        out += residual
        out = self.relu(out)
        
        return out


class Residual_Convs(nn.Module):
    def __init__(self, in_channels=1, channels=[32, 64, 160, 256]):
        super(Residual_Convs, self).__init__()

        self.channels = channels
        self.res_1 = ResBlock(in_channels, channels[0], stride=1)
        self.res_2 = ResBlock(channels[0], channels[1], stride=2)
        self.res_3 = ResBlock(channels[1], channels[2], stride=2)
        self.res_4 = ResBlock(channels[2], channels[3], stride=2)

    def forward(self, x):
        x1 = self.res_1(x)
        x2 = self.res_2(x1)
        x3 = self.res_3(x2)
        x4 = self.res_4(x3)

        return x1, x2, x3, x4
