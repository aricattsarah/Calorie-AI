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
# --- Test Time Augmentation for Validation ---
def validate_with_tta(model, val_loader, criterion, device, num_tta=3):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Standard prediction
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Test Time Augmentation for better accuracy
            if num_tta > 1:
                tta_outputs = []
                for _ in range(num_tta):
                    # Apply random augmentation
                    augmented = torch.flip(images, dims=[3]) if torch.rand(1) > 0.5 else images
                    tta_out = model(augmented)
                    tta_outputs.append(tta_out)
                
                # Average predictions
                outputs = torch.mean(torch.stack(tta_outputs), dim=0)
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, accuracy
# --- Advanced Learning Rate Scheduling ---
def get_advanced_scheduler(optimizer, train_loader, num_epochs):
    """Advanced learning rate scheduling for long training"""
    total_steps = len(train_loader) * num_epochs
    
    # OneCycleLR for the first phase, then CosineAnnealingWarmRestarts
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.001,
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=10000.0
    )
    
    return scheduler

# --- Comprehensive Plotting ---
def plot_comprehensive_history(train_losses, val_losses, train_accuracies, val_accuracies, 
                             lr_history, epoch_times, save_path='comprehensive_training.png'):
    """Plot comprehensive training analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[0, 2].plot(epochs, lr_history, 'g-', linewidth=2)
    axes[0, 2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_yscale('log')
    
    # Epoch time
    axes[1, 0].plot(epochs, epoch_times, 'purple', linewidth=2)
    axes[1, 0].set_title('Training Speed per Epoch', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy gap
    accuracy_gap = [t - v for t, v in zip(train_accuracies, val_accuracies)]
    axes[1, 1].plot(epochs, accuracy_gap, 'orange', linewidth=2)
    axes[1, 1].set_title('Training-Validation Gap', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Gap (%)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Performance summary
    axes[1, 2].text(0.1, 0.9, 'Training Summary', fontsize=16, fontweight='bold', transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.8, f'Best Val Accuracy: {max(val_accuracies):.2f}%', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.7, f'Final Val Accuracy: {val_accuracies[-1]:.2f}%', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.6, f'Final Gap: {accuracy_gap[-1]:.2f}%', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.5, f'Total Epochs: {len(epochs)}', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.4, f'Avg Time/Epoch: {np.mean(epoch_times):.2f}s', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.3, f'Total Time: {sum(epoch_times)/3600:.2f}h', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
