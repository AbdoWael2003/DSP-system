import tkinter as tk
from tkinter import ttk
class DialogButtons:
    
    def __init__(self, window,width = 900, height = 90):
        self.dialog = tk.Toplevel()
        self.mainFrame = tk.Frame(self.dialog); self.mainFrame.pack() 
        self.dialog.geometry(f"{width}x{height}+{window.winfo_screenwidth() // 2 - (width // 2)}+{window.winfo_screenheight() // 2 - (height // 2)}") 
        self.dialog.resizable(False, False)
    
class Button:
    def __init__(self, root, name, position = None, on_click = None, on_enter = None, on_leave = None, width = 35, height = 5, padx=20, pady=20): 
  
        self.button = tk.Button(root,width = width,height = height, text = name, command = on_click)
        if position is None:
            self.button.pack(padx = padx, pady = pady)
        else:
            self.button.grid(row = position[0],column = position[1],padx = padx, pady = pady)
        
        # the function first parameter must be the event
        if on_enter is not None:
            # self.button.state(['active'])
            self.button.bind("<Enter>", on_enter)
            
        # the function first parameter must be the event
        if on_leave is not None:
            # self.button.state(['!active'])
            self.button.bind("<Leave>", on_leave)