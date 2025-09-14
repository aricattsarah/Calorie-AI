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
