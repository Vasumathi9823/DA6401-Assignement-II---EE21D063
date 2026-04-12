"""Localization modules"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """Initialize the VGG11Localizer model."""
        super().__init__()
        
        self.encoder = VGG11Encoder(in_channels)
        
        # Regression head for bounding box prediction
        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            # Final output is exactly 4 continuous values
            nn.Linear(4096, 4) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model."""
        # Extract features
        x = self.encoder(x, return_features=False)
        
        # Flatten spatial dimensions
        x = torch.flatten(x, 1)
        
        # Output coordinates
        x = self.regressor(x)
        return x
