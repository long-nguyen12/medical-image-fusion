from attention.modules import CBAM
from torch import nn
import torch
from torch.nn import functional as F
from backbone.res2net import custom_res2net50_v1b


class SkipCBAMConnection(nn.Module):
    def __init__(self, f1_dim, f2_dim) -> None:
        super().__init__()

        self.cbam_1 = CBAM(f1_dim)
        self.cbam_2 = CBAM(f2_dim)
        dim = f1_dim

        self.conv = nn.Conv2d(2 * dim, dim, 1)

    def forward(self, f1, f2):
        x1 = self.cbam_1(f1)
        x2 = self.cbam_2(f2)

        out = torch.cat([x1, x2], dim=1)
        out = self.conv(out)

        return out


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = custom_res2net50_v1b()
        self.skip_1 = SkipCBAMConnection(64, 64)
        self.skip_2 = SkipCBAMConnection(128, 128)
        self.skip_3 = SkipCBAMConnection(256, 256)
        self.skip_4 = SkipCBAMConnection(512, 512)

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
        self.embed_dim = 256
        for i, dim in enumerate([64, 128, 256, 512]):
            self.add_module(f"linear_c{i+1}", MLP(dim, self.embed_dim))

        self.linear_fuse = ConvModule(self.embed_dim * 4, self.embed_dim)
        self.linear_pred = nn.Conv2d(self.embed_dim, 1, 1)
        self.dropout = nn.Dropout2d(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        features = self.encoder(x, y)
        B, _, H, W = features[0].shape

        outs = [
            self.linear_c1(features[0])
            .permute(0, 2, 1)
            .reshape(B, -1, *features[0].shape[-2:])
        ]

        for i, feature in enumerate(features[1:]):
            cf = (
                eval(f"self.linear_c{i+2}")(feature)
                .permute(0, 2, 1)
                .reshape(B, -1, *feature.shape[-2:])
            )
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


if __name__ == "__main__":
    model = FusionModel()
    x = torch.randn(1, 1, 256, 256)
    y = torch.randn(1, 1, 256, 256)
    out = model(x, y)
    print(out.shape)
