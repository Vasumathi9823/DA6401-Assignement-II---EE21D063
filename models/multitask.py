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
            gdown.download(id="16MJTJQPXdTv1FJKlU9l-KiqGgRZdfPwP", output=classifier_path, quiet=False)
            gdown.download(id="1hZsUKQIxvWwmhlvbpfNFfHcuZW8u76Ac", output=localizer_path, quiet=False)
            gdown.download(id="18LLoiujBfjrT-YW9clt7hmc6_RpJpexp", output=unet_path, quiet=False)
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
        # FORCE EVAL MODE: Protects BN running stats
        self.eval() 
        
        # ONE shared forward pass!
        bottleneck, features = self.shared_encoder(x, return_features=True)
        flat_bottleneck = torch.flatten(bottleneck, 1)
        
        # 1. Classification Branch
        class_logits = self.classifier_head(flat_bottleneck)
        
        # 2. Localization Branch (Manually applying the logic from your localization.py forward method)
        loc_logits = self.localizer_head(flat_bottleneck)
        loc_preds = torch.sigmoid(loc_logits) * 224.0
        
        # 3. Segmentation Branch (Executing the U-Net expansive path using the shared features)
        d5 = self.unet.up5(bottleneck)
        d5 = torch.cat([d5, features['pool5_pre']], dim=1)
        d5 = self.unet.dec5(d5)
        
        d4 = self.unet.up4(d5)
        d4 = torch.cat([d4, features['pool4_pre']], dim=1)
        d4 = self.unet.dec4(d4)
        
        d3 = self.unet.up3(d4)
        d3 = torch.cat([d3, features['pool3_pre']], dim=1)
        d3 = self.unet.dec3(d3)
        
        d2 = self.unet.up2(d3)
        d2 = torch.cat([d2, features['pool2_pre']], dim=1)
        d2 = self.unet.dec2(d2)
        
        d1 = self.unet.up1(d2)
        d1 = torch.cat([d1, features['pool1_pre']], dim=1)
        d1 = self.unet.dec1(d1)
        
        seg_logits = self.unet.final_conv(d1)
        
        return {
            'classification': class_logits,
            'localization': loc_preds,
            'segmentation': seg_logits
        }
