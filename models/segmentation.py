"""Segmentation model"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder

class VGG11UNet(nn.Module):
    """U-Net style segmentation network."""

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """Initialize the VGG11UNet model."""
        super().__init__()
        
        self.encoder = VGG11Encoder(in_channels)
        
        # Helper function for symmetric decoder blocks
        def dec_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Transposed Convolutions and Decoder Blocks
        # The channel size doubles due to concatenation with the encoder skip connections
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = dec_block(1024, 512) 
        
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = dec_block(512, 256) 
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = dec_block(256, 128) 
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = dec_block(128, 64) 
        
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = dec_block(128, 64) 
        
        # Final 1x1 Convolution to map to the number of classes
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model."""
        bottleneck, features = self.encoder(x, return_features=True)
        
        # Expansive Path (Decoding)
        d5 = self.up5(bottleneck)
        d5 = torch.cat([d5, features['pool5_pre']], dim=1)
        d5 = self.dec5(d5)
        
        d4 = self.up4(d5)
        d4 = torch.cat([d4, features['pool4_pre']], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, features['pool3_pre']], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, features['pool2_pre']], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, features['pool1_pre']], dim=1)
        d1 = self.dec1(d1)
        
        return self.final_conv(d1)
