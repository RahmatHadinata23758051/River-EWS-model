"""
U-Net Model Definition
Model segmentasi sederhana untuk water detection di flood images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Double Convolution Block (Conv -> BatchNorm -> ReLU) x2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block (MaxPool -> DoubleConv)"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block (ConvTranspose -> DoubleConv)"""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels)
    
    def forward(self, x1, x2):
        # x1: upsampled feature from encoder
        # x2: skip connection from decoder
        x1 = self.up(x1)
        
        # Pad if necessary to match sizes (for odd dimensions)
        if x1.size(-1) != x2.size(-1) or x1.size(-2) != x2.size(-2):
            x1 = F.pad(x1, (
                0, x2.size(-1) - x1.size(-1),
                0, x2.size(-2) - x1.size(-2)
            ))
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net Segmentation Model
    Input: (batch, 3, 256, 256) - RGB image
    Output: (batch, 1, 256, 256) - Segmentation mask (0-1)
    """
    def __init__(self, in_channels=3, out_channels=1, features=32):
        super(UNet, self).__init__()
        
        # Encoder (contracting path)
        self.inc = DoubleConv(in_channels, features)
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)
        self.down4 = Down(features * 8, features * 16)
        
        # Decoder (expanding path)
        self.up1 = Up(features * 16, features * 8)
        self.up2 = Up(features * 8, features * 4)
        self.up3 = Up(features * 4, features * 2)
        self.up4 = Up(features * 2, features)
        
        # Output layer
        self.outc = nn.Conv2d(features, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        x = self.outc(x)
        x = torch.sigmoid(x)  # Sigmoid for binary classification
        
        return x


def create_model(device='cuda', pretrained=False):
    """Create and return U-Net model"""
    model = UNet(in_channels=3, out_channels=1, features=32)
    model = model.to(device)
    return model


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = create_model(device=device)
    params = count_parameters(model)
    
    print("\n" + "="*60)
    print("U-NET MODEL ARCHITECTURE")
    print("="*60)
    print(model)
    print(f"\nTrainable Parameters: {params:,}")
    
    # Test forward pass
    test_input = torch.randn(1, 3, 256, 256).to(device)
    test_output = model(test_input)
    print(f"\nInput shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print(f"Output range: [{test_output.min():.4f}, {test_output.max():.4f}]")
