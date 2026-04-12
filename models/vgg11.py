"""VGG11 encoder"""

from typing import Dict, Tuple, Union
import torch
import torch.nn as nn

class VGG11(nn.Module):
    """Standard VGG11 topology wrapper for the autograder."""
    def __init__(self, in_channels: int = 3, num_classes: int = 1000):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, use_bn=False)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        x = self.encoder(x, return_features=False)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns."""
    def __init__(self, in_channels: int = 3, use_bn: bool = True):
        super().__init__()
        
        def conv_block(in_c, out_c):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.block1 = conv_block(in_channels, 64)
        self.block2 = conv_block(64, 128)
        self.block3 = nn.Sequential(conv_block(128, 256), conv_block(256, 256))
        self.block4 = nn.Sequential(conv_block(256, 512), conv_block(512, 512))
        self.block5 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_bn = use_bn

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
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
