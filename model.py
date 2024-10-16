from modules.modules import CBAM, BasicConv, SELayer
from torch import nn
import torch
from torch.nn import functional as F
from backbone.residual_cbam import Residual_Convs, ResBlock
from head.fpn import FPNHead
from modules.common import Conv, ConvModule


class DilationConvModule(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, d, g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# Dilation Attention Module
class DAM(nn.Module):
    def __init__(self, c1, c2, d=None, out_k=1) -> None:
        super().__init__()
        d = (1, 3, 5)

        self.conv11 = BasicConv(c1, c2, 1)

        self.d_1 = DilationConvModule(
            c2,
            c2,
            (3, 3),
            1,
            p=(1 * d[0] + 1, 1 * d[0] + 1),
            d=(d[0] + 1, d[0] + 1),
        )
        self.d_3 = DilationConvModule(
            c2,
            c2,
            (3, 3),
            1,
            p=(1 * d[1] + 1, 1 * d[1] + 1),
            d=(d[1] + 1, d[1] + 1),
        )
        self.d_5 = DilationConvModule(
            c2,
            c2,
            (3, 3),
            1,
            p=(1 * d[2] + 1, 1 * d[2] + 1),
            d=(d[2] + 1, d[2] + 1),
        )

    def forward(self, x):
        x = self.conv11(x)
        x_1 = self.d_1(x)
        x_3 = self.d_3(x)
        x_5 = self.d_5(x)
        x_atten = x_1 * x_3 * x_5
        x_out = x + x_atten

        return x_out


class FusionConnection(nn.Module):
    def __init__(self, c1, c2) -> None:
        super().__init__()

        self.cbam = CBAM(c1)
        self.cbam_2 = CBAM(c1)

        # self.conv11 = nn.Conv2d(2 * c2, c2, 1, padding=0, groups=c2)

        # self.conv1_1 = nn.Conv2d(c2, c2, (1, 3), padding=(0, 1), groups=c2)
        # self.conv1_2 = nn.Conv2d(c2, c2, (3, 1), padding=(1, 0), groups=c2)

        # self.conv2_1 = nn.Conv2d(c2, c2, (1, 5), padding=(0, 2), groups=c2)
        # self.conv2_2 = nn.Conv2d(c2, c2, (5, 1), padding=(2, 0), groups=c2)

        # self.conv3 = nn.Conv2d(c2, c2, 1)

        # self.se = SELayer(c2)

        self.conv_1 = Conv(2 * c2, c2, 3, 1, 1)
        self.conv_2 = Conv(2 * c2, c2, 5, 1, 2)

    def forward(self, x1, x2):
        x1 = self.cbam(x1)
        x2 = self.cbam(x2)

        x_cat = torch.cat([x1, x2], dim=1)

        # x_1 = self.conv11(x_cat)

        # x_3 = self.conv1_1(x_1)
        # x_3 = self.conv1_2(x_3)

        # x_5 = self.conv2_1(x_1)
        # x_5 = self.conv2_2(x_5)

        # x_sum = x_1 + x_3 + x_5
        # x_sum = self.conv3(x_sum)
        # atten = x_sum * x1 * x2
        
        x_3 = self.conv_1(x_cat)
        x_5 = self.conv_2(x_cat)
        atten_1 = x1 * x_3
        atten_2 = x2 * x_5

        atten = atten_1 + atten_2 + x1 + x2

        return atten


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = Residual_Convs(in_channels=1, channels=[32, 64, 128, 256])

        self.skip_1 = FusionConnection(32, 32)
        self.skip_2 = FusionConnection(64, 64)
        self.skip_3 = FusionConnection(128, 128)
        self.skip_4 = FusionConnection(256, 256)

        self.convs = ConvModule(2, 3, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, img_1, img_2):
        features_1 = self.encoder(img_1)
        features_2 = self.encoder(img_2)

        x1, x2, x3, x4 = features_1
        y1, y2, y3, y4 = features_2

        fm_1 = self.skip_1(x1, y1)
        fm_2 = self.skip_2(x2, y2)
        fm_3 = self.skip_3(x3, y3)
        fm_4 = self.skip_4(x4, y4)

        return fm_1, fm_2, fm_3, fm_4


class FusionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = FPNHead(self.encoder.encoder.channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        features = self.encoder(x, y)
        out = self.decoder(features)
        out = self.sigmoid(out)
        out = F.interpolate(out, size=x.size()[2:], mode="bilinear", align_corners=True)

        return out


from thop import profile
from thop import clever_format


def CalParams(model, x, y):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(x, y))
    flops, params = clever_format([flops, params], "%.3f")
    print("[Statistics Information]\nFLOPs: {}\nParams: {}".format(flops, params))


if __name__ == "__main__":
    model = FusionModel()
    x = torch.randn(1, 1, 256, 256)
    y = torch.randn(1, 1, 256, 256)
    CalParams(model, x, y)
