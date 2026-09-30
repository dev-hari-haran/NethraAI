import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class VesselUNet(nn.Module):
    """
    Lightweight, high-resolution U-Net for retinal blood vessel extraction.
    Input: [B, 1, H, W] (Green channel + CLAHE) or [B, 3, H, W] (RGB)
    Output: [B, 1, H, W] vessel probability map in range [0, 1]
    """
    def __init__(self, in_channels=1, base_features=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_features)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features, base_features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features * 2, base_features * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features * 4, base_features * 8))
        
        self.up1 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_features * 8, base_features * 4)
        
        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_features * 4, base_features * 2)
        
        self.up3 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_features * 2, base_features)
        
        self.outc = nn.Conv2d(base_features, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder with Skip Connections
        u1 = self.up1(x4)
        if u1.shape != x3.shape:
            u1 = F.interpolate(u1, size=x3.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.conv_up1(torch.cat([x3, u1], dim=1))
        
        u2 = self.up2(d1)
        if u2.shape != x2.shape:
            u2 = F.interpolate(u2, size=x2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.conv_up2(torch.cat([x2, u2], dim=1))
        
        u3 = self.up3(d2)
        if u3.shape != x1.shape:
            u3 = F.interpolate(u3, size=x1.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.conv_up3(torch.cat([x1, u3], dim=1))
        
        logits = self.outc(d3)
        probs = torch.sigmoid(logits)
        return probs

if __name__ == "__main__":
    model = VesselUNet(in_channels=1)
    dummy = torch.randn(2, 1, 384, 384)
    out = model(dummy)
    print("VesselUNet initialized successfully. Output shape:", out.shape)
