from attention.modules import CBAM
from torch import nn
import torch
from torch.nn import functional as F
from backbone.residual_cbam import Residual_Convs, ResBlock
from attention.mit import MiT
from head.fpn import FPNHead
from backbone.res2net import res2net50_26w_4s, res2net50


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


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, bias=False)
        self.bn = nn.BatchNorm2d(c2)  # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class ConvolutionalAttention(nn.Module):

    def __init__(self, dim):
        super(ConvolutionalAttention, self).__init__()
        # input
        self.conv55 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        self.conv17_0 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv17_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv111_0 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv111_1 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv211_0 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv211_1 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv11 = nn.Conv2d(dim, dim, 1)  # channel mixer

    def forward(self, x):

        skip = x.clone()

        c55 = self.conv55(x)
        c17 = self.conv17_0(x)
        c17 = self.conv17_1(c17)
        c111 = self.conv111_0(x)
        c111 = self.conv111_1(c111)
        c211 = self.conv211_0(x)
        c211 = self.conv211_1(c211)

        add = c55 + c17 + c111 + c211

        mixer = self.conv11(add)

        op = mixer * skip

        return op


class FusionConnection(nn.Module):
    def __init__(self, c1, c2) -> None:
        super().__init__()

        self.cbam_1 = CBAM(c1)
        self.cbam_2 = CBAM(c1)
        self.att_1 = ConvolutionalAttention(c1)
        # self.att_2 = MiT(c1, c2)

        d = (1, 2, 3)
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

        self.conv = ConvModule(len(d) * c2, c2)

    def forward(self, x1, x2):
        x1 = self.cbam_1(x1)
        x2 = self.cbam_2(x2)

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

        # added = x1 + x2
        
        # out = added + out

        # att = self.att_1(out)

        # out = att + out
        # out = self.mit(x)
        # print(out.shape, x1.shape, x2.shape)

        return out


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = res2net50(pretrained=False)
        self.skip_1 = FusionConnection(256, 256)
        self.skip_2 = FusionConnection(512, 512)
        self.skip_3 = FusionConnection(1024, 1024)
        self.skip_4 = FusionConnection(2048, 2048)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.convs = ConvModule(1, 3, 1)

    def forward(self, img_1, img_2):
        img_1 = self.convs(img_1)
        img_2 = self.convs(img_2)

        features_1 = self.encoder(img_1)
        features_2 = self.encoder(img_2)

        x1, x2, x3, x4 = features_1
        y1, y2, y3, y4 = features_2

        skip_mod_1 = self.skip_1(x1, y1)
        skip_mod_2 = self.skip_2(x2, y2)
        skip_mod_3 = self.skip_3(x3, y3)
        skip_mod_4 = self.skip_4(x4, y4)

        return skip_mod_1, skip_mod_2, skip_mod_3, skip_mod_4


class FusionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = FPNHead([256, 512, 1024, 2048])
        self.embed_dim = 64
        self.dim = 32

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        features = self.encoder(x, y)
        out = self.decoder(features)
        out = self.sigmoid(out)
        out = F.interpolate(out, size=x.size()[2:], mode="bicubic", align_corners=True)

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
