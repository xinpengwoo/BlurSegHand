""" Full assembly of the parts to form the complete network """
import torch.nn as nn
import models.modules.unet_parts as parts


class MiniUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = parts.DoubleConv(n_channels, 512)
        self.down1 = parts.Down(512, 1024)
        factor = 2 if bilinear else 1
        self.down3 = parts.Down(1024, 2048 // factor)

        self.up1 = parts.Up(2048, 1024 // factor, bilinear)
        self.up3 = parts.Up(1024, 2048, bilinear)
        self.outc = parts.OutConv(2048, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x4 = self.down3(x2)
        x = self.up1(x4, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits