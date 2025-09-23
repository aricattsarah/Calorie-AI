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
