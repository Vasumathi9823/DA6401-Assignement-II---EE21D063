"""Dataset skeleton for Oxford-IIIT Pet."""

import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader."""

    def __init__(self, root_dir: str, split: str = "train", transforms=None):
        """
        Args:
            root_dir: Path to the extracted Oxford-IIIT Pet dataset.
            split: 'train' for training set, 'test' for validation/test set.
            transforms: Albumentations compose object.
        """
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "annotations", "trimaps")
        self.xml_dir = os.path.join(root_dir, "annotations", "xmls")
        
        # Determine the split file to read
        split_file_name = "trainval.txt" if split == "train" else "test.txt"
        split_file = os.path.join(root_dir, "annotations", split_file_name)
        
        self.samples = []
        with open(split_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    # Convert class label to 0-indexed (0 to 36)
                    class_id = int(parts[1]) - 1 
                    self.samples.append({"img_name": img_name, "class_id": class_id})

    def _get_bounding_box_voc(self, img_name: str, img_width: int, img_height: int):
        """Fetches bounding box in [xmin, ymin, xmax, ymax] format."""
        xml_path = os.path.join(self.xml_dir, f"{img_name}.xml")
        
        # Fallback if no bounding box XML exists for a specific image
        if not os.path.exists(xml_path):
            return [0.0, 0.0, float(img_width), float(img_height)]
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find("object/bndbox")
        
        if bndbox is not None:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            return [xmin, ymin, xmax, ymax]
        else:
            return [0.0, 0.0, float(img_width), float(img_height)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_name = sample["img_name"]
        
        # 1. Load Image
        img_path = os.path.join(self.images_dir, f"{img_name}.jpg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        # 2. Load Label
        label = sample["class_id"]
        
        # 3. Load Mask (Trimap)
        mask_path = os.path.join(self.masks_dir, f"{img_name}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Oxford Pet trimaps have values 1, 2, 3. 
        # Shift to 0, 1, 2 for PyTorch segmentation losses.
        mask = mask - 1 
        
        # 4. Load Bounding Box
        bbox_voc = self._get_bounding_box_voc(img_name, w, h)
        
        # 5. Apply Albumentations Transforms
        if self.transforms is not None:
            # Pass bounding boxes in VOC format so Albumentations can adjust them during resizing
            transformed = self.transforms(
                image=image, 
                mask=mask, 
                bboxes=[bbox_voc], 
                class_labels=[label]
            )
            image = transformed["image"]
            mask = transformed["mask"]
            if len(transformed["bboxes"]) > 0:
                bbox_voc = transformed["bboxes"][0]
                
        # 6. Convert VOC format [xmin, ymin, xmax, ymax] to [cx, cy, w, h]
        xmin, ymin, xmax, ymax = bbox_voc
        box_w = xmax - xmin
        box_h = ymax - ymin
        cx = xmin + (box_w / 2.0)
        cy = ymin + (box_h / 2.0)
        final_bbox = [cx, cy, box_w, box_h]

        # 7. Convert to Tensors
        # Albumentations ToTensorV2 usually handles image transposition, but adding a fallback just in case
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
            
        bbox_tensor = torch.tensor(final_bbox, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "image": image,
            "label": label_tensor,
            "bbox": bbox_tensor,
            "mask": mask
        }
