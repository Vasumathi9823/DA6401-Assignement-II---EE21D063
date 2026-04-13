"""Localization modules"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.0):
        """Initialize the VGG11Localizer model."""
        super().__init__()
        
        self.encoder = VGG11Encoder(in_channels)
        
        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model."""
        x = self.encoder(x, return_features=False)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        
        return torch.sigmoid(x)
