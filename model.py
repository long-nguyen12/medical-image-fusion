from attention.modules import CBAM
from torch import nn
import torch
from torch.nn import functional as F
from backbone.res2net import custom_res2net50_v1b
from head.fpn import FPNHead


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


class FusionConnection(nn.Module):
    def __init__(self, c1, c2, scales=(1, 2, 3)) -> None:
        super().__init__()

        d = scales
        self.d_1 = DilationConvModule(
            c1,
            c2,
            (3, 3),
            1,
            p=(1 * d[0] + 1, 1 * d[0] + 1),
            d=(d[0] + 1, d[0] + 1),
            g=c2,
        )
        self.d_2 = DilationConvModule(
            c1,
            c2,
            (3, 3),
            1,
            p=(1 * d[1] + 1, 1 * d[1] + 1),
            d=(d[1] + 1, d[1] + 1),
            g=c2,
        )
        self.d_3 = DilationConvModule(
            c1,
            c2,
            (3, 3),
            1,
            p=(1 * d[2] + 1, 1 * d[2] + 1),
            d=(d[2] + 1, d[2] + 1),
            g=c2,
        )

        self.conv = ConvModule(c1 * 2 + len(scales) * c2, c2)

    def forward(self, x1, x2):

        x1_d_1 = self.d_1(x1)

        x1_d_2 = self.d_2(x1)

        x1_d_3 = self.d_3(x1)

        x2_d_1 = self.d_1(x2)

        x2_d_2 = self.d_2(x2)

        x2_d_3 = self.d_3(x2)

        xd_1 = x1_d_1 + x2_d_1
        xd_3 = x1_d_2 + x2_d_2
        xd_5 = x1_d_3 + x2_d_3

        out = torch.cat([x1, x2, xd_1, xd_3, xd_5], dim=1)
        out = self.conv(out)

        return out


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = custom_res2net50_v1b()
        self.skip_1 = FusionConnection(32, 32)
        self.skip_2 = FusionConnection(64, 32)
        self.skip_3 = FusionConnection(128, 32)
        self.skip_4 = FusionConnection(256, 32)

    def forward(self, img_1, img_2):
        features_1 = self.encoder(img_1)
        features_2 = self.encoder(img_2)

        x1, x2, x3, x4 = features_1
        y1, y2, y3, y4 = features_2

        con_1 = self.skip_1(x1, y1)
        con_2 = self.skip_2(x2, y2)
        con_3 = self.skip_3(x3, y3)
        con_4 = self.skip_4(x4, y4)
        return con_1, con_2, con_3, con_4


class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)  # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class FusionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = FPNHead([32, 32, 32, 32], 64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        features = self.encoder(x, y)
        out = self.decoder(features)
        out = self.sigmoid(out)
        out = F.interpolate(
            out, size=x.size()[2:], mode="bilinear", align_corners=False
        )

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
