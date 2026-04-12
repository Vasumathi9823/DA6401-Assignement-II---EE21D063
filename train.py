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

# Import custom modules
from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss

# Global variable to store 3rd layer activations for W&B Report
activation_cache = {}

def get_activation(name):
    """Hook function to capture intermediate activations."""
    def hook(model, input, output):
        activation_cache[name] = output.detach()
    return hook

def get_dataloaders(root_dir: str, batch_size: int = 32):
    """Configures Albumentations transforms and PyTorch DataLoaders."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    train_dataset = OxfordIIITPetDataset(root_dir, split="train", transforms=train_transform)
    val_dataset = OxfordIIITPetDataset(root_dir, split="test", transforms=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader

def train_classifier(args, device, train_loader, val_loader):
    """Training loop for Task 1: Classification."""
    run_name = f"classifier_bn_{args.use_bn}_drop_{args.dropout}"
    wandb.init(project="DA6401_Assignment II", name=run_name, config=vars(args))

    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout, use_bn=args.use_bn).to(device)
    
    target_layer = model.encoder.block3[0][0] if args.use_bn else model.encoder.block3[0]
    target_layer.register_forward_hook(get_activation('conv3_activations'))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images, labels = batch['image'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images, labels = batch['image'].to(device), batch['label'].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if batch_idx == 0 and 'conv3_activations' in activation_cache:
                    wandb.log({"conv3_activations": wandb.Histogram(activation_cache['conv3_activations'].cpu().numpy())}, commit=False)

        val_loss /= len(val_loader)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_macro_f1": macro_f1})
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Macro F1: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "best_metric": best_f1}, "checkpoints/classifier.pth")

    wandb.finish()

def train_localization(args, device, train_loader, val_loader):
    """Training loop for Task 2: Object Localization."""
    wandb.init(project="DA6401_Assignment II", name="task2_localization", config=vars(args))

    model = VGG11Localizer(in_channels=3, dropout_p=args.dropout).to(device)
    
    # Combined Loss: MSE + Custom IoU
    criterion_mse = nn.MSELoss()
    criterion_iou = IoULoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images, bboxes = batch['image'].to(device), batch['bbox'].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            loss_mse = criterion_mse(outputs, bboxes)
            loss_iou = criterion_iou(outputs, bboxes)
            loss = loss_mse + loss_iou
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss, val_iou_loss = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, bboxes = batch['image'].to(device), batch['bbox'].to(device)
                outputs = model(images)
                
                l_mse = criterion_mse(outputs, bboxes)
                l_iou = criterion_iou(outputs, bboxes)
                val_loss += (l_mse + l_iou).item()
                val_iou_loss += l_iou.item()

        val_loss /= len(val_loader)
        avg_val_iou = 1.0 - (val_iou_loss / len(val_loader)) # Convert loss back to IoU score

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_iou": avg_val_iou})
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {avg_val_iou:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "best_metric": best_loss}, "checkpoints/localizer.pth")

    wandb.finish()

def train_segmentation(args, device, train_loader, val_loader):
    """Training loop for Task 3: U-Net Semantic Segmentation."""
    wandb.init(project="DA6401_Assignment II", name="task3_segmentation", config=vars(args))

    model = VGG11UNet(num_classes=3, in_channels=3, dropout_p=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_dice = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images, masks = batch['image'].to(device), batch['mask'].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss, dice_score = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, masks = batch['image'].to(device), batch['mask'].to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Basic Dice Score Calculation
                preds = torch.argmax(outputs, dim=1)
                for c in range(3):
                    pred_c = (preds == c)
                    mask_c = (masks == c)
                    intersection = (pred_c & mask_c).sum().float()
                    union = pred_c.sum() + mask_c.sum()
                    dice_score += (2. * intersection / (union + 1e-6)).item()
                    
        val_loss /= len(val_loader)
        dice_score /= (len(val_loader) * 3) # Average over batches and 3 classes

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_dice": dice_score})
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {dice_score:.4f}")

        if dice_score > best_dice:
            best_dice = dice_score
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "best_metric": best_dice}, "checkpoints/unet.pth")

    wandb.finish()

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "localization", "segmentation"])
    parser.add_argument("--data_dir", type=str, default="/content/oxford-iiit-pet") 
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--use_bn", type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs("checkpoints", exist_ok=True)
    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size)

    print(f"Executing Task: {args.task} on {device}")

    if args.task == "classification":
        train_classifier(args, device, train_loader, val_loader)
    elif args.task == "localization":
        train_localization(args, device, train_loader, val_loader)
    elif args.task == "segmentation":
        train_segmentation(args, device, train_loader, val_loader)
