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
# --- Main High-Capacity Training ---
def main():
    print("🚀 INITIATING HIGH-CAPACITY TRAINING")
    print("⚡ Target: ~300MB model, 8+ hours training, maximum accuracy")
    print("=" * 70)
    
    optimize_performance()
    result = check_hardware()
    
    if result[0] is None:
        print("❌ GPU required for high-capacity training!")
        return
    
    batch_size, model_type, use_gpu = result
    
    # HIGH-CAPACITY TRAINING PARAMETERS
    data_dir = r"C:\Users\arjun\.vscode\Raghav AI\Raghav AI\Database\ImageScraper\imgs_new"
    num_epochs = 150  # Much longer training
    base_learning_rate = 0.001
    weight_decay = 0.0001  # Lighter weight decay for high capacity
    
    device = torch.device("cuda")
    print(f"Using device: {device}")
    
    # Aggressive data loading for speed
    num_workers = min(8, psutil.cpu_count())
    pin_memory = True
    
    # Advanced transforms
    train_transform, val_transform = get_advanced_transforms()
    
    # Load datasets
    print("📁 Loading datasets with advanced augmentation...")
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)
    
    num_classes = len(train_dataset.classes)
    print(f"🎯 Found {num_classes} classes: {train_dataset.classes}")
    
    # Split dataset
    train_size = int(0.85 * len(train_dataset))  # More training data
    val_size = len(train_dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(train_dataset)), [train_size, val_size]
    )
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    
    # High-performance data loaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=True, prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=True, prefetch_factor=2
    )
    
    print(f"📊 Training samples: {len(train_subset)}")
    print(f"📊 Validation samples: {len(val_subset)}")
    print(f"📊 Batch size: {batch_size}")
    print(f"📊 Workers: {num_workers}")
    
    # High-capacity model
    print(f"🏗️  Building high-capacity model...")
    model = HighCapacityFoodClassifier(num_classes, model_type)
    model = model.to(device)
   # Model size calculation
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    print(f"🔢 Total parameters: {total_params:,}")
    print(f"🔢 Trainable parameters: {trainable_params:,}")
    print(f"🔢 Estimated model size: {model_size_mb:.1f} MB")
    
    # Advanced optimizer and loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=base_learning_rate, 
                           weight_decay=weight_decay, betas=(0.9, 0.999))
    
    # Advanced scheduler
    scheduler = get_advanced_scheduler(optimizer, train_loader, num_epochs)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    lr_history = []
    epoch_times = []
    
    print("\n🎯 STARTING HIGH-CAPACITY TRAINING")
    print("=" * 70)
    print(f"🎯 Target epochs: {num_epochs}")
    print(f"🎯 Expected duration: ~8 hours")
    print(f"🎯 Target model size: ~300MB")
    print("=" * 70)
    
    start_time = time.time()
    best_val_acc = 0.0
    training_start = datetime.now()
for epoch in range(num_epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        print(f"\n🔄 Epoch [{epoch+1}/{num_epochs}] | LR: {current_lr:.8f}")
        
        # Training
        train_loss, train_acc, train_time = train_epoch_advanced(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )
        
        # Validation with TTA
        val_loss, val_acc = validate_with_tta(model, val_loader, criterion, device, num_tta=3)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # Progress update
        elapsed_hours = (time.time() - start_time) / 3600
        eta_hours = (elapsed_hours / (epoch + 1)) * (num_epochs - epoch - 1)
        
        print(f"📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"📈 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"⏱️  Epoch time: {epoch_time:.2f}s | Elapsed: {elapsed_hours:.2f}h | ETA: {eta_hours:.2f}h")
        
        # Step scheduler
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_acc': train_acc,
                'val_acc': val_acc
            }, 'best_high_capacity_model.pth')
            print(f"🏆 NEW BEST MODEL! Val Acc: {val_acc:.2f}%")
        
        # Checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'lr_history': lr_history
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
# Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    # Training complete
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("🎉 HIGH-CAPACITY TRAINING COMPLETE!")
    print("=" * 70)
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"🏆 Final validation accuracy: {val_accuracies[-1]:.2f}%")
    print(f"⏱️  Total training time: {total_time/3600:.2f} hours")
    print(f"📊 Final model size: ~{model_size_mb:.1f} MB")
    
    # Save final model
    final_model_path = "high_capacity_food_classifier.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'num_classes': num_classes,
        'class_names': train_dataset.classes,
        'best_val_acc': best_val_acc,
        'training_time_hours': total_time/3600,
        'model_size_mb': model_size_mb,
        'final_val_acc': val_accuracies[-1]
    }, final_model_path)
    
    print(f"💾 Final model saved: {final_model_path}")
    
    # Generate comprehensive plots
    print("📊 Generating comprehensive training analysis...")
    plot_comprehensive_history(train_losses, val_losses, train_accuracies, 
                              val_accuracies, lr_history, epoch_times)
    
    # Save training log
    training_log = {
        'training_duration_hours': total_time/3600,
        'best_val_accuracy': best_val_acc,
        'final_val_accuracy': val_accuracies[-1],
        'model_size_mb': model_size_mb,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'epochs_completed': len(train_losses),
        'model_type': model_type,
        'batch_size': batch_size,
        'num_classes': num_classes,
        'training_start': training_start.isoformat(),
        'training_end': datetime.now().isoformat()
    }
    
    with open('high_capacity_training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print("✅ Training log saved: high_capacity_training_log.json")
    print(f"🎯 MISSION ACCOMPLISHED: {best_val_acc:.2f}% accuracy in {total_time/3600:.2f} hours!")

if __name__ == "__main__":
    main()
