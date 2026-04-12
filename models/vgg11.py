"""VGG11 encoder"""

from typing import Dict, Tuple, Union
import torch
import torch.nn as nn

class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns."""

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super().__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Standard VGG11 Configuration
        self.block1 = conv_block(in_channels, 64)
        self.block2 = conv_block(64, 128)
        self.block3 = nn.Sequential(conv_block(128, 256), conv_block(256, 256))
        self.block4 = nn.Sequential(conv_block(256, 512), conv_block(512, 512))
        self.block5 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass."""
        features = {}
        
        x = self.block1(x)
        if return_features: features['pool1_pre'] = x
        x = self.pool(x)
        
        x = self.block2(x)
        if return_features: features['pool2_pre'] = x
        x = self.pool(x)
        
        x = self.block3(x)
        if return_features: features['pool3_pre'] = x
        x = self.pool(x)
        
        x = self.block4(x)
        if return_features: features['pool4_pre'] = x
        x = self.pool(x)
        
        x = self.block5(x)
        if return_features: features['pool5_pre'] = x
        x = self.pool(x)
        
        if return_features:
            return x, features
        return x
