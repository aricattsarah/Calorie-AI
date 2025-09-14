import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import time
import psutil
import gc
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import json
from datetime import datetime
# --- High-Capacity Model Architecture ---
class HighCapacityFoodClassifier(nn.Module):
    def __init__(self, num_classes, model_type="efficientnet_b4"):
        super(HighCapacityFoodClassifier, self).__init__()
        
        print(f"🏗️  Building high-capacity model: {model_type}")
        
        # Use larger, more powerful models
        if model_type == "efficientnet_b4":
            self.backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
            backbone_features = 1792
        elif model_type == "efficientnet_b3":
            self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            backbone_features = 1536
        elif model_type == "efficientnet_b2":
            self.backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
            backbone_features = 1408
        else:
            # ResNet alternative for comparison
            self.backbone = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
            backbone_features = 2048
        
        # Fine-tune more layers for higher capacity
        # Only freeze first 50% of layers
        total_params = list(self.backbone.parameters())
        freeze_until = len(total_params) // 2
        
        for i, param in enumerate(total_params):
            if i < freeze_until:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # High-capacity classifier head
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(backbone_features, 2048),
                nn.ReLU(),
                nn.BatchNorm1d(2048),
                nn.Dropout(0.4),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
        else:  # ResNet
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(backbone_features, 2048),
                nn.ReLU(),
                nn.BatchNorm1d(2048),
                nn.Dropout(0.4),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, x):
        return self.backbone(x)
# --- Advanced Data Augmentation ---
def get_advanced_transforms():
    """Advanced augmentation strategy for high-accuracy training"""
    
    # Training augmentations - more aggressive
    train_transform = transforms.Compose([
        transforms.Resize((288, 288)),  # Larger input size
        transforms.RandomResizedCrop(256, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])
    
    # Validation with Test Time Augmentation (TTA)
    val_transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# --- Advanced Training Function ---
def train_epoch_advanced(model, train_loader, criterion, optimizer, device, scaler, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Label smoothing effect
            if epoch < 10:  # Apply stronger regularization early
                loss = loss * 1.1
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # More frequent progress updates for long training
        if batch_idx % 5 == 0:
            elapsed = time.time() - start_time
            batches_per_sec = (batch_idx + 1) / elapsed
            eta = (len(train_loader) - batch_idx) / batches_per_sec if batches_per_sec > 0 else 0
            current_acc = 100 * correct / total
            print(f'Batch [{batch_idx:4d}/{len(train_loader)}] | Loss: {loss.item():.4f} | Acc: {current_acc:.2f}% | Speed: {batches_per_sec:.1f} b/s | ETA: {eta:.0f}s')
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    epoch_time = time.time() - start_time
    
    return avg_loss, accuracy, epoch_time
