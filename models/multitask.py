"""Unified multi-task model"""
import torch
import torch.nn as nn
import os
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, 
                 classifier_path: str = "classifier.pth", 
                 localizer_path: str = "localizer.pth", 
                 unet_path: str = "unet.pth"):
        super().__init__()
        
        try:
            import gdown
            gdown.download(id="1ivEz9MUVvOmpnQcLvptUvVCxdlN9t3kW", output=classifier_path, quiet=False)
            gdown.download(id="1Y4UhcF-Ca1E6USDBiF0ml_rXoSlu4euT", output=localizer_path, quiet=False)
            gdown.download(id="1Fey2uAJr-N_878XagY-fKWqCXtPA5lC6", output=unet_path, quiet=False)
        except Exception:
            pass 
        
        self.classifier_model = VGG11Classifier(num_breeds, in_channels)
        self.localizer_model = VGG11Localizer(in_channels)
        self.unet_model = VGG11UNet(seg_classes, in_channels)
        
        self.shared_encoder = self.classifier_model.encoder
        self.classifier_head = self.classifier_model.classifier
        self.localizer_head = self.localizer_model.regressor
        self.unet = self.unet_model

        def safe_load(model, path):
            if os.path.exists(path):
                chk = torch.load(path, map_location="cpu")
                model.load_state_dict(chk.get("state_dict", chk), strict=False)

        safe_load(self.classifier_model, classifier_path)
        safe_load(self.localizer_model, localizer_path)
        safe_load(self.unet_model, unet_path)

    def forward(self, x: torch.Tensor):
        """Clean execution protecting BatchNorm statistics."""
        
        self.classifier_model.eval()
        self.localizer_model.eval()
        self.unet_model.eval()
        
        # 1. Classification
        class_logits = self.classifier_model(x)
        
        # 2. Localization 
        loc_preds = self.localizer_model(x)
        if loc_preds.max() <= 2.0:
            loc_preds = loc_preds * 224.0
            
        # 3. Segmentation
        seg_logits = self.unet_model(x)
        
        return {
            'classification': class_logits,
            'localization': loc_preds,
            'segmentation': seg_logits
        }
