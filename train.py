# """Training entrypoint"""

"""Training entrypoint"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
from sklearn.metrics import f1_score
import numpy as np

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss

activation_cache = {}
def get_activation(name):
    def hook(model, input, output):
        activation_cache[name] = output.detach()
    return hook

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def get_dataloaders(root_dir: str, batch_size: int = 32):
    mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225)
    train_transform = A.Compose([
        A.Resize(224, 224), A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.Normalize(mean=mean, std=std), ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    val_transform = A.Compose([
        A.Resize(224, 224), A.Normalize(mean=mean, std=std), ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    train_loader = DataLoader(OxfordIIITPetDataset(root_dir, split="train", transforms=train_transform), batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(OxfordIIITPetDataset(root_dir, split="test", transforms=val_transform), batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader

def voc_to_cxcywh(bboxes):
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    return torch.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dim=1)

def cxcywh_to_voc(bboxes):
    cx, cy, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    return torch.stack([cx - (w / 2.0), cy - (h / 2.0), cx + (w / 2.0), cy + (h / 2.0)], dim=1)

def draw_bboxes_wandb(images, pred_boxes, target_boxes, ious):
    """Helper for Section 2.5: Log images with bounding boxes to W&B"""
    wandb_images = []
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(images.device)
    
    for i in range(min(10, images.size(0))):
        img = images[i] * std + mean
        img = torch.clamp(img, 0, 1).cpu().numpy().transpose(1, 2, 0)
        
        p_x1, p_y1, p_x2, p_y2 = pred_boxes[i].cpu().numpy()
        t_x1, t_y1, t_x2, t_y2 = target_boxes[i].cpu().numpy()
        iou_val = ious[i].item()
        
        wandb_images.append(wandb.Image(img, boxes={
            "predictions": {"box_data": [{"position": {"minX": p_x1, "minY": p_y1, "maxX": p_x2, "maxY": p_y2}, "class_id": 1, "domain": "pixel"}], "class_labels": {1: f"Pred IoU:{iou_val:.2f}"}},
            "ground_truth": {"box_data": [{"position": {"minX": t_x1, "minY": t_y1, "maxX": t_x2, "maxY": t_y2}, "class_id": 2, "domain": "pixel"}], "class_labels": {2: "Target"}}
        }))
    return wandb_images

# === TASK 1: CLASSIFICATION ===
def train_classifier(args, device, train_loader, val_loader):
    wandb.init(project="DA6401_Assignment II", name=f"classifier_bn_{args.use_bn}_drop_{args.dropout}", config=vars(args))
    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout, use_bn=args.use_bn).to(device)
    model.apply(init_weights)
    
    target_layer = model.encoder.block3[0][0] if args.use_bn else model.encoder.block3[0]
    target_layer.register_forward_hook(get_activation('conv3_activations'))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    best_f1 = 0.0; epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train(); train_loss = 0.0
        for batch in train_loader:
            images, labels = batch['image'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval(); val_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images, labels = batch['image'].to(device), batch['label'].to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                if batch_idx == 0 and 'conv3_activations' in activation_cache:
                    wandb.log({"conv3_activations": wandb.Histogram(activation_cache['conv3_activations'].cpu().numpy())}, commit=False)

        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        scheduler.step(macro_f1)
        wandb.log({"epoch": epoch, "train_loss": train_loss/len(train_loader), "val_loss": val_loss/len(val_loader), "val_macro_f1": macro_f1})
        print(f"Epoch {epoch+1} - Macro F1: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1; epochs_no_improve = 0
            torch.save({"state_dict": model.state_dict()}, "checkpoints/classifier.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 15: break
    wandb.finish()

# === TASK 2: LOCALIZATION ===
def train_localization(args, device, train_loader, val_loader):
    wandb.init(project="DA6401_Assignment II", name="task2_localization", config=vars(args))
    model = VGG11Localizer(in_channels=3).to(device)
    model.apply(init_weights) 
    
    criterion_reg = nn.SmoothL1Loss(); criterion_iou = IoULoss(reduction="none")
    # Use a much smaller learning rate (1e-5 instead of 5e-5)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

    # Increase patience from 5 to 15 so it doesn't drop the LR prematurely
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)
    best_iou = 0.0; epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train(); train_loss = 0.0
        for batch in train_loader:
            images = batch['image'].to(device)
            bboxes_xyxy = batch['bbox'].to(device) 
            bboxes_cxcywh = voc_to_cxcywh(bboxes_xyxy)
            
            optimizer.zero_grad()
            outputs_cxcywh = model(images)
            outputs_xyxy = cxcywh_to_voc(outputs_cxcywh)
            
            loss = criterion_reg(outputs_cxcywh, bboxes_cxcywh) + (50.0 * criterion_iou(outputs_xyxy, bboxes_xyxy).mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval(); val_loss, val_iou_loss = 0.0, 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images = batch['image'].to(device)
                bboxes_xyxy = batch['bbox'].to(device)
                bboxes_cxcywh = voc_to_cxcywh(bboxes_xyxy)
                
                outputs_cxcywh = model(images)
                outputs_xyxy = cxcywh_to_voc(outputs_cxcywh)
                
                l_iou_batch = criterion_iou(outputs_xyxy, bboxes_xyxy)
                l_iou = l_iou_batch.mean()
                val_loss += (criterion_reg(outputs_cxcywh, bboxes_cxcywh) + (50.0 * l_iou)).item()
                val_iou_loss += l_iou.item()

                # W&B SECTION 2.5: Log 10 Bounding Box images on the first batch of validation
                if batch_idx == 0:
                    ious_scores = 1.0 - l_iou_batch
                    wandb.log({"Localization_Edge_Cases": draw_bboxes_wandb(images, outputs_xyxy, bboxes_xyxy, ious_scores)}, commit=False)

        avg_val_iou = 1.0 - (val_iou_loss / len(val_loader))
        scheduler.step(avg_val_iou)
        wandb.log({"epoch": epoch, "train_loss": train_loss/len(train_loader), "val_loss": val_loss/len(val_loader), "val_iou": avg_val_iou})
        print(f"Epoch {epoch+1} - Val IoU: {avg_val_iou:.4f}")

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou; epochs_no_improve = 0
            torch.save({"state_dict": model.state_dict()}, "checkpoints/localizer.pth")
        # else:
        #     epochs_no_improve += 1
        #     if epochs_no_improve >= 15: break
    wandb.finish()

# === TASK 3: SEGMENTATION ===
def train_segmentation(args, device, train_loader, val_loader):
    wandb.init(project="DA6401_Assignment II", name=f"task3_segmentation_{args.freeze_mode}", config=vars(args))
    model = VGG11UNet(num_classes=3, in_channels=3, dropout_p=args.dropout).to(device)
    model.apply(init_weights)

    # W&B SECTION 2.3: Transfer Learning Freeze Modes
    if args.freeze_mode == "frozen":
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif args.freeze_mode == "partial":
        for name, param in model.encoder.named_parameters():
            if "block5" not in name: 
                param.requires_grad = False
    
    class_weights = torch.tensor([1.0, 0.2, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    best_dice = 0.0; epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train(); train_loss = 0.0
        for batch in train_loader:
            images, masks = batch['image'].to(device), batch['mask'].to(device, dtype=torch.long)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval(); val_loss, dice_score, pixel_acc = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, masks = batch['image'].to(device), batch['mask'].to(device, dtype=torch.long)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()

                preds = torch.argmax(outputs, dim=1)
                
                # W&B SECTION 2.6: Pixel Accuracy Calculation
                pixel_acc += (preds == masks).sum().float().item() / masks.numel()

                for c in range(3):
                    pred_c = (preds == c)
                    mask_c = (masks == c)
                    intersection = (pred_c & mask_c).sum().float()
                    union = pred_c.sum() + mask_c.sum()
                    dice_score += (2. * intersection / (union + 1e-6)).item()
                    
        dice_score /= (len(val_loader) * 3)
        pixel_acc /= len(val_loader)
        
        scheduler.step(dice_score)
        wandb.log({"epoch": epoch, "train_loss": train_loss/len(train_loader), "val_loss": val_loss/len(val_loader), "val_dice": dice_score, "val_pixel_acc": pixel_acc})
        print(f"Epoch {epoch+1} - Val Dice: {dice_score:.4f} | Val Pixel Acc: {pixel_acc:.4f}")

        if dice_score > best_dice:
            best_dice = dice_score; epochs_no_improve = 0
            torch.save({"state_dict": model.state_dict()}, "checkpoints/unet.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 15: break
    wandb.finish()

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="classification")
    parser.add_argument("--data_dir", type=str, default="/content/oxford-iiit-pet") 
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--use_bn", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--freeze_mode", type=str, default="none", choices=["none", "frozen", "partial"])

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)
    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size)

    if args.task == "classification": train_classifier(args, device, train_loader, val_loader)
    elif args.task == "localization": train_localization(args, device, train_loader, val_loader)
    elif args.task == "segmentation": train_segmentation(args, device, train_loader, val_loader)

# import os
# import argparse
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import wandb
# from sklearn.metrics import f1_score
# import numpy as np

# from data.pets_dataset import OxfordIIITPetDataset
# from models.classification import VGG11Classifier
# from models.localization import VGG11Localizer
# from models.segmentation import VGG11UNet
# from losses.iou_loss import IoULoss

# activation_cache = {}
# def get_activation(name):
#     def hook(model, input, output):
#         activation_cache[name] = output.detach()
#     return hook

# def init_weights(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, 0, 0.01)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)

# def get_dataloaders(root_dir: str, batch_size: int = 32):
#     mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225)
#     train_transform = A.Compose([
#         A.Resize(224, 224), A.HorizontalFlip(p=0.5),
#         A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
#         A.Normalize(mean=mean, std=std), ToTensorV2()
#     ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

#     val_transform = A.Compose([
#         A.Resize(224, 224), A.Normalize(mean=mean, std=std), ToTensorV2()
#     ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

#     train_loader = DataLoader(OxfordIIITPetDataset(root_dir, split="train", transforms=train_transform), batch_size=batch_size, shuffle=True, num_workers=2)
#     val_loader = DataLoader(OxfordIIITPetDataset(root_dir, split="test", transforms=val_transform), batch_size=batch_size, shuffle=False, num_workers=2)
#     return train_loader, val_loader

# def voc_to_cxcywh(bboxes):
#     x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
#     return torch.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dim=1)

# def cxcywh_to_voc(bboxes):
#     cx, cy, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
#     return torch.stack([cx - (w / 2.0), cy - (h / 2.0), cx + (w / 2.0), cy + (h / 2.0)], dim=1)

# # === TASK 1: CLASSIFICATION ===
# def train_classifier(args, device, train_loader, val_loader):
#     wandb.init(project="DA6401_Assignment II", name=f"scratch_classifier_bn_{args.use_bn}_drop_{args.dropout}", config=vars(args))
#     model = VGG11Classifier(num_classes=37, dropout_p=args.dropout, use_bn=args.use_bn).to(device)
#     model.apply(init_weights)
    
#     target_layer = model.encoder.block3[0][0] if args.use_bn else model.encoder.block3[0]
#     target_layer.register_forward_hook(get_activation('conv3_activations'))
    
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
#     best_f1 = 0.0; epochs_no_improve = 0

#     for epoch in range(args.epochs):
#         model.train(); train_loss = 0.0
#         for batch in train_loader:
#             images, labels = batch['image'].to(device), batch['label'].to(device)
#             optimizer.zero_grad()
#             loss = criterion(model(images), labels)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             train_loss += loss.item()

#         model.eval(); val_loss, all_preds, all_labels = 0.0, [], []
#         with torch.no_grad():
#             for batch_idx, batch in enumerate(val_loader):
#                 images, labels = batch['image'].to(device), batch['label'].to(device)
#                 outputs = model(images)
#                 val_loss += criterion(outputs, labels).item()
#                 all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
#                 if batch_idx == 0 and 'conv3_activations' in activation_cache:
#                     wandb.log({"conv3_activations": wandb.Histogram(activation_cache['conv3_activations'].cpu().numpy())}, commit=False)

#         macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
#         scheduler.step(macro_f1)
#         wandb.log({"epoch": epoch, "train_loss": train_loss/len(train_loader), "val_loss": val_loss/len(val_loader), "val_macro_f1": macro_f1})
#         print(f"Epoch {epoch+1} - Macro F1: {macro_f1:.4f}")

#         if macro_f1 > best_f1:
#             best_f1 = macro_f1; epochs_no_improve = 0
#             torch.save({"state_dict": model.state_dict()}, "checkpoints/classifier.pth")
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= 15: break
#     wandb.finish()

# # === TASK 2: LOCALIZATION ===
# def train_localization(args, device, train_loader, val_loader):
#     wandb.init(project="DA6401_Assignment II", name="scratch_task2_localization", config=vars(args))
#     model = VGG11Localizer(in_channels=3).to(device)
#     model.apply(init_weights) 
    
#     criterion_reg = nn.SmoothL1Loss(); criterion_iou = IoULoss(reduction="mean")
#     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
#     best_iou = 0.0; epochs_no_improve = 0

#     for epoch in range(args.epochs):
#         model.train(); train_loss = 0.0
#         for batch in train_loader:
#             images = batch['image'].to(device)
#             bboxes_xyxy = batch['bbox'].to(device) 
#             bboxes_cxcywh = voc_to_cxcywh(bboxes_xyxy)
            
#             optimizer.zero_grad()
#             outputs_cxcywh = model(images)
#             outputs_xyxy = cxcywh_to_voc(outputs_cxcywh)
            
#             loss = criterion_reg(outputs_cxcywh, bboxes_cxcywh) + (50.0 * criterion_iou(outputs_xyxy, bboxes_xyxy))
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             train_loss += loss.item()

#         model.eval(); val_loss, val_iou_loss = 0.0, 0.0
#         with torch.no_grad():
#             for batch in val_loader:
#                 images = batch['image'].to(device)
#                 bboxes_xyxy = batch['bbox'].to(device)
#                 bboxes_cxcywh = voc_to_cxcywh(bboxes_xyxy)
                
#                 outputs_cxcywh = model(images)
#                 outputs_xyxy = cxcywh_to_voc(outputs_cxcywh)
                
#                 l_iou = criterion_iou(outputs_xyxy, bboxes_xyxy)
#                 val_loss += (criterion_reg(outputs_cxcywh, bboxes_cxcywh) + (50.0 * l_iou)).item()
#                 val_iou_loss += l_iou.item()
                
#         avg_val_iou = 1.0 - (val_iou_loss / len(val_loader))
#         scheduler.step(avg_val_iou)
#         wandb.log({"epoch": epoch, "train_loss": train_loss/len(train_loader), "val_loss": val_loss/len(val_loader), "val_iou": avg_val_iou})
#         print(f"Epoch {epoch+1} - Val IoU: {avg_val_iou:.4f}")

#         if avg_val_iou > best_iou:
#             best_iou = avg_val_iou; epochs_no_improve = 0
#             torch.save({"state_dict": model.state_dict()}, "checkpoints/localizer.pth")
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= 15: break
#     wandb.finish()

# # === TASK 3: SEGMENTATION ===
# def train_segmentation(args, device, train_loader, val_loader):
#     wandb.init(project="DA6401_Assignment II", name="scratch_task3_segmentation", config=vars(args))
#     model = VGG11UNet(num_classes=3, in_channels=3, dropout_p=args.dropout).to(device)
#     model.apply(init_weights)
    
#     # Class weights to fix the "Predict 100% Background" bug
#     class_weights = torch.tensor([1.0, 0.2, 1.0]).to(device)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
#     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
#     best_dice = 0.0; epochs_no_improve = 0

#     for epoch in range(args.epochs):
#         model.train(); train_loss = 0.0
#         for batch in train_loader:
#             images, masks = batch['image'].to(device), batch['mask'].to(device, dtype=torch.long)
#             optimizer.zero_grad()
#             loss = criterion(model(images), masks)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             train_loss += loss.item()

#         model.eval(); val_loss, dice_score = 0.0, 0.0
#         with torch.no_grad():
#             for batch in val_loader:
#                 images, masks = batch['image'].to(device), batch['mask'].to(device, dtype=torch.long)
#                 outputs = model(images)
#                 val_loss += criterion(outputs, masks).item()

#                 preds = torch.argmax(outputs, dim=1)
#                 for c in range(3):
#                     pred_c = (preds == c)
#                     mask_c = (masks == c)
#                     intersection = (pred_c & mask_c).sum().float()
#                     union = pred_c.sum() + mask_c.sum()
#                     dice_score += (2. * intersection / (union + 1e-6)).item()
                    
#         dice_score /= (len(val_loader) * 3)
#         scheduler.step(dice_score)
#         wandb.log({"epoch": epoch, "train_loss": train_loss/len(train_loader), "val_loss": val_loss/len(val_loader), "val_dice": dice_score})
#         print(f"Epoch {epoch+1} - Val Dice: {dice_score:.4f}")

#         if dice_score > best_dice:
#             best_dice = dice_score; epochs_no_improve = 0
#             torch.save({"state_dict": model.state_dict()}, "checkpoints/unet.pth")
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= 15: break
#     wandb.finish()

# def str2bool(v):
#     if isinstance(v, bool): return v
#     if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
#     return False

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--task", type=str, default="classification")
#     parser.add_argument("--data_dir", type=str, default="/content/oxford-iiit-pet") 
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--lr", type=float, default=1e-4)
#     parser.add_argument("--epochs", type=int, default=100)
#     parser.add_argument("--dropout", type=float, default=0.5)
#     parser.add_argument("--use_bn", type=str2bool, nargs='?', const=True, default=True)

#     args = parser.parse_args()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.makedirs("checkpoints", exist_ok=True)
#     train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size)

#     if args.task == "classification": train_classifier(args, device, train_loader, val_loader)
#     elif args.task == "localization": train_localization(args, device, train_loader, val_loader)
#     elif args.task == "segmentation": train_segmentation(args, device, train_loader, val_loader)
