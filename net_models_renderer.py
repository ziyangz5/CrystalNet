import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

# UNet code is from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode = 'reflect'),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding_mode = 'reflect')

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        factor = 2 if bilinear else 1
        self.down4 = (Down(256, 512 // factor))
        self.up1 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64 // factor, bilinear))
        self.up4 = (Up(64, 32, bilinear))
        self.outc = (OutConv(32, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    
class RNet(nn.Module):
    def __init__(self,in_channel):
        super(RNet, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=5,padding=2, padding_mode = 'reflect'),
            nn.LeakyReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3,padding=1, padding_mode = 'reflect'),
            nn.LeakyReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3,padding=1, padding_mode = 'reflect'),
            nn.LeakyReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, padding_mode = 'reflect'),
            #nn.Sigmoid()
        )
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    
class TNetBackBone(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(TNetBackBone, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3,padding=1, padding_mode = 'reflect'),
            nn.LeakyReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, out_channel, kernel_size=3,padding=1, padding_mode = 'reflect'),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
class TNet(nn.Module):
    def __init__(self,in_channel,hidden=16):
        super(TNet, self).__init__()
        self.tnet_f = TNetBackBone(in_channel,hidden)
        self.tnet_b = TNetBackBone(hidden,8)
        self.hidden = hidden
    def forward(self, g,rx):
        num_of_glass = g.shape[1]
        r_shape = (g.shape[0],self.hidden,g.shape[-2],g.shape[-1])
        result = torch.zeros(r_shape,device="cuda")
        for i in range(num_of_glass):
            result = result + self.tnet_f(torch.cat([rx, g[:,i,...]], 1) )
        result = self.tnet_b(result)
        return result
def fc_layer(in_features, out_features):
    net = nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LeakyReLU(inplace=True)
    )

    return net

class CrystalRenderer(nn.Module):
    def __init__(self):
        super(CrystalRenderer, self).__init__()
        self.inner_pos = fc_layer(in_features=3, out_features=512)
        self.unet = UNet(27+512,8)
        self.tnet = TNet(8 + 17)
        self.r_unet = UNet(16+27,8)
        self.rnet = RNet(8+27+8)
    def forward(self, x,g):
        position = x[:,4:7, :, :]
        position = torch.moveaxis(position, 1, 3)
        
        pe = self.inner_pos(position)
        pe = torch.moveaxis(pe, 3, 1)
        x0 = torch.cat([x, pe], 1)
        rx = self.unet(x0)
        tx = self.tnet(g,rx)
        rtx = torch.cat([rx, tx, x], 1)
        embed_rtx = self.r_unet(rtx)
        result = self.rnet(torch.cat([embed_rtx,x,tx], 1))
        return result