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
        
       # Image preview
        self.image_label = ttk.Label(input_frame, text="No image selected", 
                                    background="lightgray", relief="sunken")
        self.image_label.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Analyze button
        self.analyze_btn = ttk.Button(main_frame, text="🚀 Analyze Food", 
                                     command=self.analyze_food, state="disabled")
        self.analyze_btn.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Results Section
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)
       # Count display
        self.count_frame = ttk.Frame(results_frame)
        self.count_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.count_frame.columnconfigure(1, weight=1)
        
        ttk.Label(self.count_frame, text="Count:", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky=tk.W)
        self.count_label = ttk.Label(self.count_frame, text="-", font=('Arial', 16, 'bold'), 
                                    foreground="blue")
        self.count_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, width=70)
        self.results_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bind events
        self.api_key.trace('w', self.check_inputs)
        self.food_name.trace('w', self.check_inputs)
        
    def browse_image(self):
        """Open file dialog to select image."""
        file_path = filedialog.askopenfilename(
            title="Select Food Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.uploaded_image_path = file_path
            self.load_image_preview(file_path)
            self.check_inputs()
    
    def load_image_preview(self, file_path):
        """Load and display image preview."""
        try:
            # Open and resize image for preview
            image = Image.open(file_path)
            image.thumbnail((300, 200), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
            # Store original image
            self.uploaded_image = Image.open(file_path)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def check_inputs(self, *args):
        """Check if all required inputs are provided."""
        can_analyze = (self.api_key.get().strip() and 
                      self.food_name.get().strip() and 
                      self.uploaded_image is not None)
        
        self.analyze_btn.configure(state="normal" if can_analyze else "disabled")
    
    def analyze_food(self):
        """Analyze the uploaded food image."""
        if not self.api_key.get().strip():
            messagebox.showerror("Error", "Please enter your Gemini API key.")
            return
        
        if not self.food_name.get().strip():
            messagebox.showerror("Error", "Please enter the food item name.")
            return
        
        if self.uploaded_image is None:
            messagebox.showerror("Error", "Please upload an image.")
            return
        
        # Start analysis in a separate thread
        self.start_analysis()
 def start_analysis(self):
        """Start the analysis process."""
        self.analyze_btn.configure(state="disabled")
        self.progress.start()
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Analyzing image...\n")
        
        # Run analysis in thread to prevent GUI freezing
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()
    
    def run_analysis(self):
        """Run the actual analysis."""
        try:
            # Configure Gemini API
            genai.configure(api_key=self.api_key.get().strip())
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Create prompt
            prompt = f"""Analyze this image of {self.food_name.get().strip()} and provide:
1. Count how many individual {self.food_name.get().strip()} items are visible in the image
2. Estimate the total nutrition information for all items combined
