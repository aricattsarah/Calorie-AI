import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageTk
import numpy as np
import json
from datetime import datetime
import threading
import queue
class HighCapacityFoodClassifier(nn.Module):
    def __init__(self, num_classes, model_type="efficientnet_b3"):
        super(HighCapacityFoodClassifier, self).__init__()
        
        self.model_type = model_type
        try:
            if model_type.startswith('efficientnet'):
                self.backbone = getattr(models, model_type)(weights=None)
                backbone_features = {
                    'efficientnet_b0': 1280,
                    'efficientnet_b1': 1280,
                    'efficientnet_b2': 1408,
                    'efficientnet_b3': 1536,
                    'efficientnet_b4': 1792
                }.get(model_type, 1536)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        except Exception as e:
            raise RuntimeError(f"Backbone initialization failed: {str(e)}")

        try:
            if hasattr(self.backbone, 'classifier'):
                in_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = self._build_classifier(in_features, num_classes)
            elif hasattr(self.backbone, 'fc'):
                in_features = self.backbone.fc.in_features
                self.backbone.fc = self._build_classifier(in_features, num_classes)
            else:
                raise AttributeError("Backbone has no classifier/fc layer")
        except Exception as e:
            raise RuntimeError(f"Classifier build failed: {str(e)}")
    def _build_classifier(self, in_features, num_classes):
        return nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 2048),
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

class FoodClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🍔 Food Classifier - AI Image Analysis")
        self.root.geometry("900x750")
        self.root.configure(bg='#f0f0f0')
        
        self.model = None
        self.class_names = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_image_path = None
        self.transform = None
        self.model_loaded = False
        self.default_model_path = r"C:\Users\arjun\.vscode\Raghav AI\Raghav AI\AI-model\Calorie AI\high_capacity_food_classifier.pth"
        self.result_queue = queue.Queue()
        
        self.setup_ui()
        self.setup_transforms()
        
        if os.path.exists(self.default_model_path):
            self.load_model_async(self.default_model_path)
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        title_label = ttk.Label(main_frame, text="🍔 AI Food Classifier", 
                               font=('Arial', 24, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        model_frame = ttk.LabelFrame(main_frame, text="Model Configuration", padding="10")
        model_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(1, weight=1)
        
        ttk.Label(model_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.model_path_var = tk.StringVar(value=self.default_model_path)
        self.model_path_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, width=60)
        self.model_path_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
