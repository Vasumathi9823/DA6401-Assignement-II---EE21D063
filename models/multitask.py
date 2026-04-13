"""Unified multi-task model"""

import torch
import torch.nn as nn
import os
from .vgg11 import VGG11Encoder
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, 
                 classifier_path: str = "classifier.pth", 
                 localizer_path: str = "localizer.pth", 
                 unet_path: str = "unet.pth"):
        """Initialize the shared backbone/heads using trained weights."""
        super().__init__()
        #https://drive.google.com/file/d/1QYn6W2AEeJ316b9cxf7hvhX8CXuK_Hzg/view?usp=sharing
                     #https://drive.google.com/file/d/1X5T2Bu7G-l8oQnJh36z37H6Pf0Q1KTem/view?usp=sharing
        #https://drive.google.com/file/d/1qHJAEFyrXF8ochAfRCyavVL7sNVVzAtE/view?usp=sharing
                     #https://drive.google.com/file/d/1HOfRw2GL8t0S1DEFyNuKUfYvGMWpJdUg/view?usp=sharing
                     #1484W3kYuCSE3MjaWNCDNuE3Uv-m86r2A
        import gdown
        gdown.download(id="1X5T2Bu7G-l8oQnJh36z37H6Pf0Q1KTem", output=classifier_path, quiet=False)
        gdown.download(id="1qHJAEFyrXF8ochAfRCyavVL7sNVVzAtE", output=localizer_path, quiet=False)
        gdown.download(id="1HOfRw2GL8t0S1DEFyNuKUfYvGMWpJdUg", output=unet_path, quiet=False)
        
        # Shared Encoder
        self.shared_encoder = VGG11Encoder(in_channels)
        
        # Task Heads
        dummy_classifier = VGG11Classifier(num_breeds, in_channels)
        dummy_localizer = VGG11Localizer(in_channels)
        self.unet = VGG11UNet(seg_classes, in_channels)
        
        self.classifier_head = dummy_classifier.classifier
        self.localizer_head = dummy_localizer.regressor
        
        # Attempt to load trained weights
        try:
            if os.path.exists(classifier_path):
                c_weights = torch.load(classifier_path, map_location="cpu")
                c_sd = c_weights.get("state_dict", c_weights)
                
                # Load backbone weights from the classifier
                encoder_sd = {k.replace("encoder.", ""): v for k, v in c_sd.items() if "encoder." in k}
                self.shared_encoder.load_state_dict(encoder_sd)
                
                # Load classifier head
                class_head_sd = {k.replace("classifier.", ""): v for k, v in c_sd.items() if "classifier." in k}
                self.classifier_head.load_state_dict(class_head_sd)

            if os.path.exists(localizer_path):
                l_weights = torch.load(localizer_path, map_location="cpu")
                l_sd = l_weights.get("state_dict", l_weights)
                loc_head_sd = {k.replace("regressor.", ""): v for k, v in l_sd.items() if "regressor." in k}
                self.localizer_head.load_state_dict(loc_head_sd)

            if os.path.exists(unet_path):
                u_weights = torch.load(unet_path, map_location="cpu")
                u_sd = u_weights.get("state_dict", u_weights)
                self.unet.load_state_dict(u_sd)

        except Exception as e:
            print(f"Warning: Could not fully load checkpoints. Error: {e}")
            
        # Point the UNet's encoder to the shared encoder memory block
        self.unet.encoder = self.shared_encoder

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model."""
        # 1. Shared Feature Extraction
        bottleneck, features = self.shared_encoder(x, return_features=True)
        flat_features = torch.flatten(bottleneck, 1)
        
        # 2. Execute Task Heads
        class_logits = self.classifier_head(flat_features)
        bbox_preds = self.localizer_head(flat_features)
        
        # 3. Segmentation Decoder (bypassing the encoder since features are already extracted)
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
            'localization': bbox_preds,
            'segmentation': seg_logits
        }
