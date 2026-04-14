"""Reusable custom layers"""

import torch
import torch.nn as nn

class CustomDropout(nn.Module):
    def __init__(self, p: float = 0.5):
      
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout probability must be between 0 and 1.")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        
        mask = (torch.rand_like(x) > self.p).float()
        
        # Apply mask and scale (Inverted Dropout)
        return x * mask / (1.0 - self.p)
