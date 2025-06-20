# unet_model1.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------ MC Dropout Layer ------------------
class MCDropout(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, x, mc_inference=False):
        return F.dropout2d(x, self.p, training=mc_inference)

# ------------------ Safe Identity Layer ------------------
class IdentityWithMC(nn.Module):
    def forward(self, x, mc_inference=False):
        return x

# ------------------ Double Convolution Block ------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, p=0.2):
        super().__init__()
        self.use_dropout = dropout
        self.dropout_layer = MCDropout(p=p) if dropout else IdentityWithMC()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, mc_inference=False):
        x = self.double_conv(x)
        x = self.dropout_layer(x, mc_inference=mc_inference)
        return x

# ------------------ U-Net Model ------------------
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, use_mc_dropout=True):
        super().__init__()
        self.use_mc_dropout = use_mc_dropout

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64, dropout=False)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128, dropout=use_mc_dropout, p=0.1)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256, dropout=use_mc_dropout, p=0.2)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512, dropout=use_mc_dropout, p=0.2)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024, dropout=use_mc_dropout, p=0.3)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512, dropout=use_mc_dropout, p=0.2)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256, dropout=use_mc_dropout, p=0.2)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128, dropout=use_mc_dropout, p=0.1)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64, dropout=use_mc_dropout, p=0.1)

        # Output layer
        self.final_conv = nn.Conv2d(64, out_channels, 1)

        self._init_weights()

    def forward(self, x, mc_inference=False):
        enc1 = self.enc1(x, mc_inference)
        enc2 = self.enc2(self.pool1(enc1), mc_inference)
        enc3 = self.enc3(self.pool2(enc2), mc_inference)
        enc4 = self.enc4(self.pool3(enc3), mc_inference)

        bottleneck = self.bottleneck(self.pool4(enc4), mc_inference)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4, mc_inference)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3, mc_inference)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2, mc_inference)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1, mc_inference)

        return self.final_conv(dec1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
