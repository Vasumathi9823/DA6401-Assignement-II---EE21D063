"""Classification components"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout

class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5, use_bn: bool = True):
        """Initialize the VGG11Classifier model."""
        super().__init__()
        
        self.encoder = VGG11Encoder(in_channels, use_bn=use_bn)
        
        # TA APPROVED SHRUNK FC LAYERS
        # Reduced from 4096 to prevent massive overfitting
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model."""
        x = self.encoder(x, return_features=False)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
