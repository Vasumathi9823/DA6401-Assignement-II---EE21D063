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

def get_dataloaders(root_dir: str, batch_size: int = 32):
    """Configures Albumentations transforms and PyTorch DataLoaders."""
    
    # Standard ResNet/VGG ImageNet normalization stats
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

def train_classifier(args, device):
    """Training loop specifically for Task 1: Classification."""
    
    # Initialize Weights & Biases
    wandb.init(project="da6401_assignment_2", name="task1_classification", config=vars(args))

    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size)
    
    # Instantiate the model
    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = 0.0

    # Ensure checkpoints directory exists
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        
        # Calculate Macro F1 Score as required by guidelines
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        # Log metrics to W&B
        wandb.log({
            "epoch": epoch, 
            "train_loss": train_loss, 
            "val_loss": val_loss, 
            "val_macro_f1": macro_f1
        })
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Macro F1: {macro_f1:.4f}")

        # Save checkpoint if it's the best model so far
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            checkpoint = {
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "best_metric": best_f1
            }
            torch.save(checkpoint, "checkpoints/classifier.pth")
            print(">>> Saved new best classifier checkpoint!")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 2 Training Script")
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "localization", "segmentation", "multitask"])
    # Point this to where the extracted dataset is located in Colab
    parser.add_argument("--data_dir", type=str, default="/content/oxford-iiit-pet") 
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.5)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Executing Task: {args.task} on {device}")

    if args.task == "classification":
        train_classifier(args, device)
