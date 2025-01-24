"""
Tkinter-based audio control interface with professional-grade UI elements
and proper threading considerations for real-time audio visualization.
"""

import tkinter as tk
from tkinter import ttk
import queue
import sounddevice as sd
import numpy as np
import asyncio
from typing import Optional, List, Dict, Any, Callable

class AudioInterface:
    def __init__(self, 
                 input_device_name: Optional[str] = None,
                 output_device_name: Optional[str] = None,
                 on_input_change: Optional[Callable[[str], None]] = None,
                 on_output_change: Optional[Callable[[str], None]] = None):
        """
        Initialize the Tkinter-based audio interface.
        
        Args:
            input_device_name: Initial input device name to display
            output_device_name: Initial output device name to display
            on_input_change: Callback when input device changes
            on_output_change: Callback when output device changes
        """
        # Initialize state
        self.running = True
        self.has_gui = True
        self.current_volume = 0
        self.input_device_name = input_device_name or "No device selected"
        self.output_device_name = output_device_name or "No device selected"
        self.on_input_change = on_input_change
        self.on_output_change = on_output_change
        
        # Create queues for thread-safe communication
        self.volume_queue = queue.Queue()
        self.input_device_queue = queue.Queue()
        self.output_device_queue = queue.Queue()
        
        try:
            # Initialize GUI on main thread
            self._init_gui()
            print("\nAudio interface opened successfully")
            
        except Exception as e:
            print(f"\nWarning: Could not create GUI window ({e})")
            print("Audio interface will not be displayed")
            self.running = False
            self.has_gui = False
    
    def _init_gui(self):
        """Initialize the Tkinter GUI"""
        # Create main window
        self.root = tk.Tk()
        self.root.title("Audio Control")
        self.root.geometry("400x300")
        # Configure grid
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create and configure frames
        self._create_input_device_frame()
        self._create_output_device_frame()
        self._create_volume_frame()
        
        # Set up periodic updates
        self._schedule_updates()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_input_device_frame(self):
        """Create frame for input device selection"""
        input_frame = ttk.LabelFrame(self.root, text="Input Device", padding=5)
        input_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        
        # Get list of input devices
        devices = self._get_input_devices()
        
        # Device selection dropdown
        self.input_device_var = tk.StringVar(value=self.input_device_name)
        self.input_device_combo = ttk.Combobox(
            input_frame, 
            textvariable=self.input_device_var,
            values=devices,
            state="readonly",
            width=40
        )
        self.input_device_combo.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.input_device_combo.bind('<<ComboboxSelected>>', self._on_input_device_change)

    def _create_output_device_frame(self):
        """Create frame for output device selection"""
        output_frame = ttk.LabelFrame(self.root, text="Output Device", padding=5)
        output_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        # Get list of output devices
        devices = self._get_output_devices()
        
        # Device selection dropdown
        self.output_device_var = tk.StringVar(value=self.output_device_name)
        self.output_device_combo = ttk.Combobox(
            output_frame, 
            textvariable=self.output_device_var,
            values=devices,
            state="readonly",
            width=40
        )
        self.output_device_combo.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.output_device_combo.bind('<<ComboboxSelected>>', self._on_output_device_change)
    
    def _create_volume_frame(self):
        """Create frame for volume visualization"""
        volume_frame = ttk.LabelFrame(self.root, text="Microphone Input Level", padding=5)
        volume_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        
        # Volume progress bar
        self.volume_bar = ttk.Progressbar(
            volume_frame,
            orient="horizontal",
            length=250,
            mode="determinate"
        )
        self.volume_bar.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        # Volume label
        self.volume_label = ttk.Label(volume_frame, text="0%")
        self.volume_label.grid(row=1, column=0, padx=5, pady=5)
    
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
        
        # Handle input device updates
        try:
            while True:
                device = self.input_device_queue.get_nowait()
                self.input_device_var.set(device)
                self.input_device_queue.task_done()
        except queue.Empty:
            pass

        # Handle output device updates
        try:
            while True:
                device = self.output_device_queue.get_nowait()
                self.output_device_var.set(device)
                self.output_device_queue.task_done()
        except queue.Empty:
            pass
    
    def _get_input_devices(self) -> List[str]:
        """Get list of available input devices"""
        devices = []
        try:
            for device in sd.query_devices():
                if device['max_input_channels'] > 0:
                    devices.append(device['name'])
        except Exception as e:
            print(f"Error getting input devices: {e}")
        return devices

    def _get_output_devices(self) -> List[str]:
        """Get list of available output devices"""
        devices = []
        try:
            for device in sd.query_devices():
                if device['max_output_channels'] > 0:
                    devices.append(device['name'])
        except Exception as e:
            print(f"Error getting output devices: {e}")
        return devices
    
    def _on_input_device_change(self, event):
        """Handle input device selection change"""
        new_device = self.input_device_var.get()
        print(f"\nSelected input device: {new_device}")
        self.input_device_name = new_device
        if self.on_input_change:
            self.on_input_change(new_device)

    def _on_output_device_change(self, event):
        """Handle output device selection change"""
        new_device = self.output_device_var.get()
        print(f"\nSelected output device: {new_device}")
        self.output_device_name = new_device
        if self.on_output_change:
            self.on_output_change(new_device)
    
    def _on_closing(self):
        """Handle window closing"""
        print("\nClosing audio interface window...")
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
        print("[Audio Interface] AudioInterface.close() called - start")
        if self.running and self.has_gui:
            self.running = False
            self.has_gui = False
            try:
                self.root.quit()
                self.root.destroy()
            except Exception as e:
                print(f"Error closing window: {e}")
        print("[Audio Interface] AudioInterface.close() finished")
