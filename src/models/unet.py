import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = ConvBlock(3, 32)
        self.encoder2 = ConvBlock(32, 64)
        self.encoder3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(ConvBlock(128, 128), nn.Dropout2d(0.5))
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.decoder2 = ConvBlock(128 + 64, 64)
        self.decoder1 = ConvBlock(64 + 32, 32)
        self.output = nn.Conv2d(32, 3, kernel_size=1)

        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, image):
        x1 = self.encoder1(image)
        x2 = self.encoder2(self.pool(x1))
        x3 = self.encoder3(self.pool(x2))
        x = self.bottleneck(x3)

        x = self.upsample(x)
        x = self.decoder2(torch.cat([x, x2], dim=1))
        x = self.upsample(x)
        x = self.decoder1(torch.cat([x, x1], dim=1))

        correction = self.output(x)
        return image + correction
