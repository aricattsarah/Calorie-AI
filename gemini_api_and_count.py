import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import google.generativeai as genai
from PIL import Image, ImageTk
import json
import threading
import os

class FoodAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🍎 Food Analysis App")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.api_key = tk.StringVar()
        self.food_name = tk.StringVar()
        self.uploaded_image = None
        self.uploaded_image_path = None
        
        # Create GUI
        self.create_widgets()
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🍎 Food Analysis App", 
                               font=('Arial', 24, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # API Key Section
        api_frame = ttk.LabelFrame(main_frame, text="API Configuration", padding="10")
        api_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        api_frame.columnconfigure(1, weight=1)
        
        ttk.Label(api_frame, text="Gemini API Key:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        api_entry = ttk.Entry(api_frame, textvariable=self.api_key, show="*", width=50)
        api_entry.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        ttk.Label(api_frame, text="Get your API key from Google AI Studio", 
                 foreground="blue").grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # Input Section
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding="10")
        input_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)
        
        # Food name input
        ttk.Label(input_frame, text="Food Item Name:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        food_entry = ttk.Entry(input_frame, textvariable=self.food_name, width=30)
        food_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Image upload
        ttk.Label(input_frame, text="Upload Image:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        upload_btn = ttk.Button(input_frame, text="Browse Image", command=self.browse_image)
        upload_btn.grid(row=1, column=1, sticky=tk.W)
        
