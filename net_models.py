import torch
import torch.nn as nn
import torch.nn.functional as F

# UNet code is from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

# Shared Components

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, use_leaky=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        activation = nn.LeakyReLU(inplace=True) if use_leaky else nn.ReLU(inplace=True)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            activation,
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            activation
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, use_leaky=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_leaky=use_leaky)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, use_leaky=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, use_leaky=use_leaky)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_leaky=use_leaky)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# UNet Variants

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, use_leaky=True):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 32, use_leaky=use_leaky)
        self.down1 = Down(32, 64, use_leaky=use_leaky)
        self.down2 = Down(64, 128, use_leaky=use_leaky)
        self.down3 = Down(128, 256, use_leaky=use_leaky)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor, use_leaky=use_leaky)
        self.up1 = Up(512, 256 // factor, bilinear, use_leaky=use_leaky)
        self.up2 = Up(256, 128 // factor, bilinear, use_leaky=use_leaky)
        self.up3 = Up(128, 64 // factor, bilinear, use_leaky=use_leaky)
        self.up4 = Up(64, 32, bilinear, use_leaky=use_leaky)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.inc(x), self.down1(x1), self.down2(x2), self.down3(x3), self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class ThreeWayUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ThreeWayUNet, self).__init__()
        self.unet = UNet(n_channels, 32, bilinear)
        
        self.out_normal = UNet(32, 3, bilinear)
        self.out_oi = UNet(32, n_classes, bilinear)
        self.out_uv = UNet(32, 2, bilinear)

    def forward(self, x):
        features = self.unet(x)
        normal, oi, uv = self.out_normal(features), self.out_oi(features), self.out_uv(features)
        return normal, oi, uv


# TNet and RNet

class TNet(nn.Module):
    def __init__(self, in_channel, hidden=32):
        super(TNet, self).__init__()
        self.tnet_f = TNetBackBone(in_channel, hidden)
        self.tnet_b = TNetBackBone(hidden, 16)

    def forward(self, g, rx):
        result = torch.zeros((g.size(0), self.hidden, g.size(-2), g.size(-1)), device=g.device)
        for i in range(g.shape[1]):
            result += self.tnet_f(torch.cat([rx, g[:, i, ...]], 1))
        return self.tnet_b(result)


class RNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RNet, self).__init__()
        self.layer0 = nn.Sequential(nn.Conv2d(in_channel, 32, kernel_size=5, padding=2), nn.LeakyReLU(inplace=True))
        self.layer1 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.LeakyReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.LeakyReLU(inplace=True))
        self.layer3 = nn.Conv2d(32, out_channel, kernel_size=1)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)


# Full Models

class CrystalRenderer(nn.Module):
    def __init__(self):
        super(CrystalRenderer, self).__init__()
        self.inner_pos = fc_layer(in_features=3, out_features=512)
        self.unet = UNet(27 + 512, 8)
        self.tnet = TNet(8 + 17)
        self.r_unet = UNet(16 + 27, 8)
        self.rnet = RNet(8 + 27 + 8, 3)

    def forward(self, x, g, ld):
        position = torch.moveaxis(x[:, 4:7, :, :], 1, 3)
        pe = torch.moveaxis(self.inner_pos(position), 3, 1)
        x0 = torch.cat([x, pe], 1)
        rx = self.unet(x0)
        tx = self.tnet(g, rx)
        rtx = torch.cat([rx, tx, x], 1)
        embed_rtx = self.r_unet(rtx)
        return self.rnet(torch.cat([embed_rtx, x, tx], 1))


class CrystalNet(nn.Module):
    def __init__(self, n_oi):
        super(CrystalNet, self).__init__()
        self.cnet = CNet(n_oi)

    def forward(self, x, g):
        return self.cnet(x, g)


# Helper Layers

def fc_layer(in_features, out_features):
    return nn.Sequential(nn.Linear(in_features, out_features), nn.LeakyReLU(inplace=True))

