import torch
import torch.nn as nn
from thop import profile
from torchsummary import summary

from model.FAM import FAM
from model.FEM import FEM
from model.MBM import MBM


class MyConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            MBM(in_channels, out_channels),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            MyConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = MyConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.inc = FEM(n_channels, out_planes=16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 128)
        self.up1 = Up(256, 64)
        self.up2 = Up(128, 32)
        self.up3 = Up(64, 16)
        self.up4 = Up(32, 8)
        self.outc = FAM(8, n_classes)


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
        return self.outc(x)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    summary(model, input_size=(3, 224, 224))

    input_size = (1, 3, 224, 224)
    inputs = torch.randn(input_size).to(device)
    flops, params = profile(model, inputs=(inputs,))
    print(f"FLOPs: {flops / 1e9:.3f} G")
    print(f"Parameters: {params / 1e6:.3f} M")
    out = model(inputs)
