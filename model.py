from attention.modules import CBAM
from torch import nn
import torch
from torch.nn import functional as F
from backbone.residual_cbam import Residual_CBAM_Block, ResBlock
from attention.mit import CrossMiT

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

        # d = scales
        # self.d_1 = DilationConvModule(
        #     c1,
        #     c2,
        #     (3, 3),
        #     1,
        #     p=(1 * d[0] + 1, 1 * d[0] + 1),
        #     d=(d[0] + 1, d[0] + 1),
        # )
        # self.d_2 = DilationConvModule(
        #     c1,
        #     c2,
        #     (3, 3),
        #     1,
        #     p=(1 * d[1] + 1, 1 * d[1] + 1),
        #     d=(d[1] + 1, d[1] + 1),
        # )
        # self.d_3 = DilationConvModule(
        #     c1,
        #     c2,
        #     (3, 3),
        #     1,
        #     p=(1 * d[2] + 1, 1 * d[2] + 1),
        #     d=(d[2] + 1, d[2] + 1),
        # )
        self.cbam = CBAM(c1)
        self.cross_mit1 = CrossMiT(c1, c2)
        self.cross_mit2 = CrossMiT(c1, c2)
        
        self.conv = ConvModule(2 * c2, c2)

    def forward(self, x1, x2, guided=None):
        # if guided is not None:
        #     x1 = x1 + guided
        #     x2 = x2 + guided

        # x1_d_1 = self.d_1(x1)
        # x1_d_2 = self.d_2(x1)
        # x1_d_3 = self.d_3(x1)

        # x2_d_1 = self.d_1(x2)
        # x2_d_2 = self.d_2(x2)
        # x2_d_3 = self.d_3(x2)

        # xd_1 = torch.cat([x1_d_1, x2_d_1], dim=1)
        # xd_3 = torch.cat([x1_d_2, x2_d_2], dim=1)
        # xd_5 = torch.cat([x1_d_3, x2_d_3], dim=1)

        # out = xd_1 + xd_3 + xd_5
        # out = torch.cat([xd_1, xd_3, xd_5], dim=1)
        
        x1 = self.cbam(x1)
        x2 = self.cbam(x2)
        
        functional_att = self.cross_mit1(x1, x2)
        anatomical_att = self.cross_mit2(x2, x1)
        out = torch.cat([functional_att, anatomical_att], dim=1)
        
        out = self.conv(out)

        return out


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = Residual_CBAM_Block(in_channels=1)
        self.skip_1 = FusionConnection(32, 32)
        self.skip_2 = FusionConnection(64, 64)
        self.skip_3 = FusionConnection(160, 160)
        self.skip_4 = FusionConnection(256, 256)

        self.cats = ResBlock(2, 32, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.up_1 = ConvModule(32, 64, 1)
        self.up_2 = ConvModule(64, 160, 1)
        self.up_3 = ConvModule(160, 256, 1)

    def forward(self, img_1, img_2):
        img_1 = self.maxpool(img_1)
        img_2 = self.maxpool(img_2)
        inputs = torch.cat([img_1, img_2], dim=1)
        guided_feature = self.cats(inputs)

        features_1 = self.encoder(img_1)
        features_2 = self.encoder(img_2)

        x1, x2, x3, x4 = features_1
        y1, y2, y3, y4 = features_2

        skip_mod_1 = self.skip_1(x1, y1, guided_feature)
        # _skip_mod_1 = self.maxpool(skip_mod_1)
        _skip_mod_1 = self.up_1(skip_mod_1)
        # print(_skip_mod_1.shape, x2.shape)

        skip_mod_2 = self.skip_2(x2, y2, _skip_mod_1)
        # _skip_mod_2 = self.maxpool(skip_mod_2)
        _skip_mod_2 = self.up_2(skip_mod_2)

        skip_mod_3 = self.skip_3(x3, y3, _skip_mod_2)
        # _skip_mod_3 = self.maxpool(skip_mod_3)
        _skip_mod_3 = self.up_3(skip_mod_3)

        skip_mod_4 = self.skip_4(x4, y4, _skip_mod_3)

        return skip_mod_1, skip_mod_2, skip_mod_3, skip_mod_4


class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, bias=False)
        self.bn = nn.BatchNorm2d(c2)  # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class FusionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        # self.decoder = Decoder()
        self.embed_dim = 64
        self.dim = 32

        self.linear_fuse = ConvModule(sum([32, 64, 160, 256]), self.embed_dim, 1)
        self.linear_pred = nn.Conv2d(self.embed_dim, 1, 1)
        self.dropout = nn.Dropout2d(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        features = self.encoder(x, y)
        # out = self.decoder(features)
        # return out
        B, _, H, W = features[0].shape
        outs = []

        for i, cf in enumerate(features):
            outs.append(
                F.interpolate(cf, size=(H, W), mode="bilinear", align_corners=False)
            )
        out = self.linear_fuse(torch.cat(outs[::-1], dim=1))
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
