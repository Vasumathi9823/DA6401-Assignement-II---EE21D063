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
        
        # Using the TA's exact gdown format with your extracted Drive IDs
        import gdown
        gdown.download(id="16MJTJQPXdTv1FJKlU9l-KiqGgRZdfPwP", output=classifier_path, quiet=False)
        gdown.download(id="1hZsUKQIxvWwmhlvbpfNFfHcuZW8u76Ac", output=localizer_path, quiet=False)
        gdown.download(id="18LLoiujBfjrT-YW9clt7hmc6_RpJpexp", output=unet_path, quiet=False)
        
        self.shared_encoder = VGG11Encoder(in_channels)
        dummy_classifier = VGG11Classifier(num_breeds, in_channels)
        dummy_localizer = VGG11Localizer(in_channels)
        self.unet = VGG11UNet(seg_classes, in_channels)
        
        self.classifier_head = dummy_classifier.classifier
        self.localizer_head = dummy_localizer.regressor
        
        # Load the downloaded weights
        try:
            if os.path.exists(classifier_path):
                c_weights = torch.load(classifier_path, map_location="cpu")
                c_sd = c_weights.get("state_dict", c_weights)
                encoder_sd = {k.replace("encoder.", ""): v for k, v in c_sd.items() if "encoder." in k}
                self.shared_encoder.load_state_dict(encoder_sd, strict=False)
                class_head_sd = {k.replace("classifier.", ""): v for k, v in c_sd.items() if "classifier." in k}
                self.classifier_head.load_state_dict(class_head_sd, strict=False)

            if os.path.exists(localizer_path):
                l_weights = torch.load(localizer_path, map_location="cpu")
                l_sd = l_weights.get("state_dict", l_weights)
                loc_head_sd = {k.replace("regressor.", ""): v for k, v in l_sd.items() if "regressor." in k}
                self.localizer_head.load_state_dict(loc_head_sd, strict=False)

            if os.path.exists(unet_path):
                u_weights = torch.load(unet_path, map_location="cpu")
                self.unet.load_state_dict(u_weights.get("state_dict", u_weights), strict=False)

        except Exception as e:
            print(f"Warning: Could not fully load checkpoints. Error: {e}")
            
        self.unet.encoder = self.shared_encoder

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model."""
        bottleneck, features = self.shared_encoder(x, return_features=True)
        flat_features = torch.flatten(bottleneck, 1)
        
        class_logits = self.classifier_head(flat_features)
        bbox_preds = self.localizer_head(flat_features)
        
        d5 = self.unet.dec5(torch.cat([self.unet.up5(bottleneck), features['pool5_pre']], dim=1))
        d4 = self.unet.dec4(torch.cat([self.unet.up4(d5), features['pool4_pre']], dim=1))
        d3 = self.unet.dec3(torch.cat([self.unet.up3(d4), features['pool3_pre']], dim=1))
        d2 = self.unet.dec2(torch.cat([self.unet.up2(d3), features['pool2_pre']], dim=1))
        d1 = self.unet.dec1(torch.cat([self.unet.up1(d2), features['pool1_pre']], dim=1))
        
        return {
            'classification': class_logits,
            'localization': bbox_preds,
            'segmentation': self.unet.final_conv(d1)
        }
