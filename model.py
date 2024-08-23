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
        )
        self.d_2 = DilationConvModule(
            c1,
            c2,
            (3, 3),
            1,
            p=(1 * d[1] + 1, 1 * d[1] + 1),
            d=(d[1] + 1, d[1] + 1),
        )
        self.d_3 = DilationConvModule(
            c1,
            c2,
            (3, 3),
            1,
            p=(1 * d[2] + 1, 1 * d[2] + 1),
            d=(d[2] + 1, d[2] + 1),
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
        self.skip_1 = FusionConnection(16, 32)
        self.skip_2 = FusionConnection(32, 32)
        self.skip_3 = FusionConnection(64, 32)
        self.skip_4 = FusionConnection(128, 32)

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


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, w, h = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FusionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.embed_dim = 64
        for i, dim in enumerate([32, 32, 32, 32]):
            self.add_module(f"linear_c{i+1}", MLP(dim, self.embed_dim))

        self.se = SELayer(self.embed_dim * 4)

        self.linear_fuse = ConvModule(self.embed_dim * 4, self.embed_dim)
        self.linear_pred = nn.Conv2d(self.embed_dim, 1, 1)
        self.dropout = nn.Dropout2d(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        features = self.encoder(x, y)
        B, _, H, W = features[0].shape

        outs = [self.linear_c1(features[0]).permute(0, 2, 1).reshape(B, -1, *features[0].shape[-2:])]

        for i, feature in enumerate(features[1:]):
            cf = eval(f"self.linear_c{i+2}")(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            outs.append(
                F.interpolate(cf, size=(H, W), mode="bilinear", align_corners=False)
            )
        out = self.se(torch.cat(outs[::-1], dim=1))
        out = self.linear_fuse(out)
        out = self.linear_pred(self.dropout(out))
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
