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
        
        ttk.Button(model_frame, text="Browse", command=self.browse_model).grid(row=0, column=2)
        ttk.Button(model_frame, text="Load Model", command=self.load_model_button).grid(row=0, column=3, padx=(10, 0))
        
        config_frame = ttk.Frame(model_frame)
        config_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Label(config_frame, text="Num Classes:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.num_classes_var = tk.StringVar(value="1052")
        self.num_classes_entry = ttk.Entry(config_frame, textvariable=self.num_classes_var, width=10)
        self.num_classes_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(config_frame, text="Model Type:").grid(row=0, column=2, sticky=tk.W, padx=(0, 10))
        self.model_type_var = tk.StringVar(value="efficientnet_b3")
        self.model_type_combo = ttk.Combobox(config_frame, textvariable=self.model_type_var, 
                                           values=["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", 
                                                  "efficientnet_b3", "efficientnet_b4"],
                                           state="readonly", width=15)
        self.model_type_combo.grid(row=0, column=3, sticky=tk.W)
        
        ttk.Label(config_frame, text="Class Names File:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.class_names_path_var = tk.StringVar()
        self.class_names_entry = ttk.Entry(config_frame, textvariable=self.class_names_path_var, width=30)
        self.class_names_entry.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=(0, 10), pady=(10, 0))
        
        ttk.Button(config_frame, text="Browse", command=self.browse_class_names).grid(row=1, column=3, pady=(10, 0))
        
        self.model_status_var = tk.StringVar(value="Model not loaded")
        self.model_status_label = ttk.Label(model_frame, textvariable=self.model_status_var, 
                                          foreground="red")
        self.model_status_label.grid(row=3, column=0, columnspan=4, pady=(10, 0))
        image_frame = ttk.LabelFrame(main_frame, text="Image Analysis", padding="10")
        image_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(2, weight=1)
        image_frame.rowconfigure(1, weight=1)
        
        upload_frame = ttk.Frame(image_frame)
        upload_frame.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        ttk.Button(upload_frame, text="📁 Upload Image", command=self.upload_image,
                  style='Accent.TButton').pack(side=tk.LEFT, padx=(0, 10))
        
        self.analyze_button = ttk.Button(upload_frame, text="🔍 Analyze Food", 
                                       command=self.analyze_food, state='disabled',
                                       style='Accent.TButton')
        self.analyze_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(upload_frame, text="🗑️ Clear", command=self.clear_results).pack(side=tk.LEFT)
        
        content_frame = ttk.Frame(image_frame)
        content_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        image_display_frame = ttk.LabelFrame(content_frame, text="Image Preview", padding="10")
        image_display_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        image_display_frame.columnconfigure(0, weight=1)
        image_display_frame.rowconfigure(0, weight=1)
        
        self.image_label = ttk.Label(image_display_frame, text="No image selected", 
                                   background='white', relief='sunken')
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        results_frame = ttk.LabelFrame(content_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        self.results_text = tk.Text(results_frame, height=15, width=40, wrap=tk.WORD,
                                   font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'))
    def setup_transforms(self):
        self.transform = transforms.Compose([
            transforms.Resize(256 + 32),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def browse_model(self):
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")]
        )
        if file_path:
            self.model_path_var.set(file_path)
    
    def browse_class_names(self):
        file_path = filedialog.askopenfilename(
            title="Select Class Names File",
            filetypes=[("Text Files", "*.txt"), ("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if file_path:
            self.class_names_path_var.set(file_path)
    
    def load_class_names_from_file(self, file_path):
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict):
                        if 'class_names' in data:
                            return data['class_names']
                        elif 'classes' in data:
                            return data['classes']
                        else:
                            return [data[str(i)] for i in range(len(data))]
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            print(f"Error loading class names from file: {e}")
            return None
