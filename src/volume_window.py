"""
Tkinter-based audio control window with professional-grade UI elements
and proper threading considerations for real-time audio visualization.
"""

import tkinter as tk
from tkinter import ttk
import queue
import sounddevice as sd
import numpy as np
from typing import Optional, List, Dict, Any

class VolumeWindow:
    def __init__(self, device_name: Optional[str] = None):
        """
        Initialize the Tkinter-based volume window.
        
        Args:
            device_name: Initial audio device name to display
        """
        # Initialize state
        self.running = True
        self.has_gui = True
        self.current_volume = 0
        self.device_name = device_name or "No device selected"
        
        # Create queues for thread-safe communication
        self.volume_queue = queue.Queue()
        self.device_queue = queue.Queue()
        
        try:
            # Initialize GUI on main thread
            self._init_gui()
            print("\nVolume control window opened successfully")
            
        except Exception as e:
            print(f"\nWarning: Could not create GUI window ({e})")
            print("Volume meter will not be displayed")
            self.running = False
            self.has_gui = False
    
    def _init_gui(self):
        """Initialize the Tkinter GUI"""
        # Create main window
        self.root = tk.Tk()
        self.root.title("Audio Control")
        self.root.geometry("400x300")
        self.root.configure(bg='white')  # Set window background
        
        # Configure grid
        self.root.grid_columnconfigure(0, weight=1)
        
        # Style configuration
        self.style = ttk.Style()
        self.style.configure("TProgressbar", thickness=15)
        self.style.configure("TLabelframe", background="white")
        self.style.configure("TLabelframe.Label", background="white")
        self.style.configure("TFrame", background="white")
        
        # Create and configure frames
        self._create_device_frame()
        self._create_volume_frame()
        self._create_control_frame()
        
        # Set up periodic updates
        self._schedule_updates()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_device_frame(self):
        """Create frame for device selection"""
        device_frame = ttk.LabelFrame(self.root, text="Audio Device", padding=10)
        device_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        
        # Get list of input devices
        devices = self._get_input_devices()
        
        # Device selection dropdown
        self.device_var = tk.StringVar(value=self.device_name)
        self.device_combo = ttk.Combobox(
            device_frame, 
            textvariable=self.device_var,
            values=devices,
            state="readonly",
            width=40
        )
        self.device_combo.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.device_combo.bind('<<ComboboxSelected>>', self._on_device_change)
    
    def _create_volume_frame(self):
        """Create frame for volume visualization"""
        volume_frame = ttk.LabelFrame(self.root, text="Volume Level", padding=10)
        volume_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        # Volume progress bar
        self.volume_bar = ttk.Progressbar(
            volume_frame,
            orient="horizontal",
            length=300,
            mode="determinate"
        )
        self.volume_bar.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        # Volume label
        self.volume_label = ttk.Label(volume_frame, text="0%")
        self.volume_label.grid(row=1, column=0, padx=5, pady=5)
    
    def _create_control_frame(self):
        """Create frame for control buttons"""
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding=10)
        control_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        
        # Button frame for organization
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=0, column=0, sticky="ew")
        button_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Control buttons
        self.start_button = ttk.Button(
            button_frame,
            text="Start",
            command=self._on_start
        )
        self.start_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.stop_button = ttk.Button(
            button_frame,
            text="Stop",
            command=self._on_stop,
            state="disabled"
        )
        self.stop_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    
    def _schedule_updates(self):
        """Schedule periodic UI updates"""
        if self.running:
            try:
                # Process any pending volume updates
                self._process_queued_updates()
                # Update window
                self.root.update_idletasks()
                # Schedule next update
                self.root.after(16, self._schedule_updates)  # ~60 FPS
            except Exception as e:
                print(f"Error in window update: {e}")
    
    def _process_queued_updates(self):
        """Process any queued updates for volume and device info"""
        # Handle volume updates
        try:
            while True:
                volume = self.volume_queue.get_nowait()
                self.volume_bar["value"] = volume
                self.volume_label["text"] = f"{volume}%"
                self.volume_queue.task_done()
        except queue.Empty:
            pass
        
        # Handle device updates
        try:
            while True:
                device = self.device_queue.get_nowait()
                self.device_var.set(device)
                self.device_queue.task_done()
        except queue.Empty:
            pass
    
    def _get_input_devices(self) -> List[str]:
        """Get list of available input devices"""
        devices = []
        try:
            for i, device in enumerate(sd.query_devices()):
                if device['max_input_channels'] > 0:
                    devices.append(device['name'])
        except Exception as e:
            print(f"Error getting input devices: {e}")
        return devices
    
    def _on_device_change(self, event):
        """Handle device selection change"""
        new_device = self.device_var.get()
        print(f"\nSelected device: {new_device}")
        # Notify main thread of device change
        self.device_name = new_device
    
    def _on_start(self):
        """Handle start button click"""
        self.start_button["state"] = "disabled"
        self.stop_button["state"] = "normal"
        self.device_combo["state"] = "disabled"
    
    def _on_stop(self):
        """Handle stop button click"""
        self.start_button["state"] = "normal"
        self.stop_button["state"] = "disabled"
        self.device_combo["state"] = "readonly"
    
    def _on_closing(self):
        """Handle window closing"""
        print("\nClosing volume control window...")
        self.running = False
        self.has_gui = False
        self.root.quit()
        self.root.destroy()
    
    def process_audio(self, audio_data):
        """
        Process audio and update volume display.
        Thread-safe method called from main thread.
        """
        if not self.running or not self.has_gui:
            return
            
        try:
            # Calculate RMS volume
            rms = np.sqrt(np.mean(audio_data**2))
            # Scale to reasonable volume range
            scaled_volume = min(1.0, rms * 3.0)
            volume = int(scaled_volume * 100)
            
            # Queue volume update for GUI thread
            self.volume_queue.put(volume)
            
        except Exception as e:
            print(f"Error processing audio: {e}")
    
    def update(self):
        """
        Process a single update of the GUI.
        This should be called periodically from the main thread.
        """
        if self.running and self.has_gui:
            try:
                self.root.update_idletasks()
                self.root.update()
            except Exception as e:
                print(f"Error updating GUI: {e}")
                self.running = False
                self.has_gui = False
    
    def close(self):
        """
        Close the window and clean up resources.
        """
        print("[Volume Window] VolumeWindow.close() called - start")
        if self.running and self.has_gui:
            self.running = False
            self.has_gui = False
            try:
                self.root.quit()
                self.root.destroy()
            except Exception as e:
                print(f"Error closing window: {e}")
        print("[Volume Window] VolumeWindow.close() finished")
