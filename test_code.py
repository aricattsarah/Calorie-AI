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
    
    def load_model_button(self):
        model_path = self.model_path_var.get()
        if not model_path:
            messagebox.showerror("Error", "Please select a model file")
            return
        
        if not os.path.exists(model_path):
            messagebox.showerror("Error", "Model file not found")
            return
        
        self.load_model_async(model_path)
    
    def load_model_async(self, model_path):
        self.model_status_var.set("Loading model...")
        self.model_status_label.configure(foreground="orange")
        self.progress.start()
        
        thread = threading.Thread(target=self.load_model_thread, args=(model_path,))
        thread.daemon = True
        thread.start()
        
        self.root.after(100, self.check_model_loading)
    
    def load_model_thread(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            num_classes = int(self.num_classes_var.get())
            model_type = self.model_type_var.get()
            
            class_names = None
            
            class_names_path = self.class_names_path_var.get()
            if class_names_path and os.path.exists(class_names_path):
                class_names = self.load_class_names_from_file(class_names_path)
            
            if isinstance(checkpoint, dict):
                saved_num_classes = checkpoint.get('num_classes')
                saved_class_names = checkpoint.get('class_names')
                saved_model_type = checkpoint.get('model_type', model_type)
                
                if saved_num_classes and self.num_classes_var.get() == "1052":
                    num_classes = saved_num_classes
                
                if saved_model_type and self.model_type_var.get() == "efficientnet_b3":
                    model_type = saved_model_type
                
                if class_names is None and saved_class_names:
                    class_names = saved_class_names
                
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                
                if 'model_state_dict' not in checkpoint:
                    possible_keys = ['state_dict', 'model', 'net']
                    for key in possible_keys:
                        if key in checkpoint:
                            state_dict = checkpoint[key]
                            break
                    else:
                        state_dict = checkpoint
           else:
                state_dict = checkpoint
            
            if class_names is None:
                class_names = [f"class_{i}" for i in range(num_classes)]
            
            if len(class_names) != num_classes:
                print(f"Warning: Class names length ({len(class_names)}) doesn't match num_classes ({num_classes})")
                if len(class_names) < num_classes:
                    class_names.extend([f"class_{i}" for i in range(len(class_names), num_classes)])
                else:
                    class_names = class_names[:num_classes]
            
            if num_classes == 1052 and self.num_classes_var.get() == "1052":
                inferred_classes = self.infer_num_classes(state_dict)
                if inferred_classes:
                    num_classes = inferred_classes
                    if len(class_names) != num_classes:
                        if class_names == [f"class_{i}" for i in range(len(class_names))]:
                            class_names = [f"class_{i}" for i in range(num_classes)]
                        else:
                            if len(class_names) < num_classes:
                                class_names.extend([f"class_{i}" for i in range(len(class_names), num_classes)])
                            else:
                                class_names = class_names[:num_classes]
            
            model = HighCapacityFoodClassifier(num_classes, model_type)
            
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    model.load_state_dict(state_dict, strict=False)
                    print(f"Warning: Loaded model with size mismatches: {e}")
                else:
                    raise e
            model = model.to(self.device)
            model.eval()
            
            self.result_queue.put(('success', model, class_names, num_classes, model_type))
            
        except Exception as e:
            self.result_queue.put(('error', str(e)))
    
    def infer_num_classes(self, state_dict):
        try:
            final_layer_patterns = [
                'backbone.classifier.6.weight',
                'backbone.classifier.6.bias',
                'backbone.fc.weight',
                'backbone.fc.bias',
                'classifier.6.weight',
                'classifier.6.bias',
                'fc.weight',
                'fc.bias'
            ]
            
            for pattern in final_layer_patterns:
                if pattern in state_dict:
                    shape = state_dict[pattern].shape
                    if 'weight' in pattern and len(shape) >= 2:
                        return shape[0]
                    elif 'bias' in pattern and len(shape) >= 1:
                        return shape[0]
            
            for key in reversed(list(state_dict.keys())):
                if 'weight' in key and len(state_dict[key].shape) >= 2:
                    if any(word in key.lower() for word in ['classifier', 'fc', 'linear']):
                        return state_dict[key].shape[0]
            
            return None
            
        except Exception as e:
            print(f"Error inferring num_classes: {e}")
            return None
    def check_model_loading(self):
        try:
            result = self.result_queue.get_nowait()
            self.progress.stop()
            
            if result[0] == 'success':
                self.model = result[1]
                self.class_names = result[2]
                num_classes = result[3]
                model_type = result[4]
                self.model_loaded = True
                
                self.num_classes_var.set(str(num_classes))
                self.model_type_var.set(model_type)
                
                self.model_status_var.set(f"Model loaded successfully! Classes: {len(self.class_names)}, Type: {model_type}")
                self.model_status_label.configure(foreground="green")
                self.status_var.set("Model loaded - Ready for analysis")
                
                if self.current_image_path:
                    self.analyze_button.configure(state='normal')
                    
            else:
                self.model_status_var.set(f"Failed to load model: {result[1]}")
                self.model_status_label.configure(foreground="red")
                self.status_var.set("Model loading failed")
                
        except queue.Empty:
            self.root.after(100, self.check_model_loading)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            try:
                image = Image.open(file_path)
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                display_size = (300, 300)
                image.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(image)
                
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo
                
                self.current_image_path = file_path
                
                if self.model_loaded:
                    self.analyze_button.configure(state='normal')
                
                filename = os.path.basename(file_path)
                self.status_var.set(f"Image loaded: {filename}")
                
                self.results_text.delete(1.0, tk.END)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    def analyze_food(self):
        if not self.model_loaded:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        if not self.current_image_path:
            messagebox.showerror("Error", "Please upload an image first")
            return
        
        self.analyze_button.configure(state='disabled')
        self.progress.start()
        self.status_var.set("Analyzing image...")
        
        thread = threading.Thread(target=self.analyze_image_thread)
        thread.daemon = True
        thread.start()
        
        self.root.after(100, self.check_analysis_complete)
    
    def analyze_image_thread(self):
        try:
            image = Image.open(self.current_image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                confidence, predicted_idx = torch.max(probabilities, 1)
                predicted_class = self.class_names[predicted_idx.item()]
                confidence_score = confidence.item()
                
                top5_probs, top5_indices = torch.topk(probabilities, min(5, len(self.class_names)))
                top5_results = [
                    (self.class_names[idx.item()], prob.item()) 
                    for idx, prob in zip(top5_indices[0], top5_probs[0])
                ]
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'top5': top5_results,
                'image_path': self.current_image_path
            }
            
            self.result_queue.put(('analysis_success', result))
            
        except Exception as e:
            self.result_queue.put(('analysis_error', str(e)))
    
    def check_analysis_complete(self):
        try:
            result = self.result_queue.get_nowait()
            self.progress.stop()
            self.analyze_button.configure(state='normal')
            
            if result[0] == 'analysis_success':
                self.display_results(result[1])
                self.status_var.set("Analysis complete")
            else:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, f"❌ Analysis failed: {result[1]}")
                self.status_var.set("Analysis failed")
