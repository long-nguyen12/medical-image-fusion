import torch
from torch import nn, Tensor
from torch.nn import functional as F


class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True),
        )


class FPNHead(nn.Module):
    def __init__(self, in_channels, channel=128, num_classes=1):
        super().__init__()
        self.lateral_convs = nn.ModuleList([])
        self.output_convs = nn.ModuleList([])

        for ch in in_channels[::-1]:
            self.lateral_convs.append(ConvModule(ch, channel, 1))
            self.output_convs.append(ConvModule(channel, channel, 1, 1, 0))

        self.conv_seg = nn.Conv2d(channel, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features) -> Tensor:
        features = features[::-1]
        out = self.lateral_convs[0](features[0])

        for i in range(1, len(features)):
            out = F.interpolate(out, scale_factor=2.0, mode="nearest")
            lat = self.lateral_convs[i](features[i])
            # out = torch.cat([out, lat], dim = 1)
            out = out + lat
            out = self.output_convs[i](out)
        out = self.conv_seg(self.dropout(out))
        return out
