"""Custom IoU loss"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression."""

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("Reduction must be 'mean', 'sum', or 'none'.")
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes."""
        # Convert from (cx, cy, w, h) to boundary coordinates (xmin, ymin, xmax, ymax)
        pred_x_min = pred_boxes[:, 0] - (pred_boxes[:, 2] / 2)
        pred_y_min = pred_boxes[:, 1] - (pred_boxes[:, 3] / 2)
        pred_x_max = pred_boxes[:, 0] + (pred_boxes[:, 2] / 2)
        pred_y_max = pred_boxes[:, 1] + (pred_boxes[:, 3] / 2)

        target_x_min = target_boxes[:, 0] - (target_boxes[:, 2] / 2)
        target_y_min = target_boxes[:, 1] - (target_boxes[:, 3] / 2)
        target_x_max = target_boxes[:, 0] + (target_boxes[:, 2] / 2)
        target_y_max = target_boxes[:, 1] + (target_boxes[:, 3] / 2)

        # Calculate Intersection boundaries
        inter_x_min = torch.max(pred_x_min, target_x_min)
        inter_y_min = torch.max(pred_y_min, target_y_min)
        inter_x_max = torch.min(pred_x_max, target_x_max)
        inter_y_max = torch.min(pred_y_max, target_y_max)

        # Calculate Intersection area (clamp to 0 to avoid negative areas when boxes don't overlap)
        inter_w = torch.clamp(inter_x_max - inter_x_min, min=0)
        inter_h = torch.clamp(inter_y_max - inter_y_min, min=0)
        intersection = inter_w * inter_h

        # Calculate Union area
        pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
        target_area = target_boxes[:, 2] * target_boxes[:, 3]
        union = pred_area + target_area - intersection

        # Compute IoU
        iou = intersection / (union + self.eps)
        
        # Loss is bounded between [0, 1]
        loss = 1.0 - iou

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
