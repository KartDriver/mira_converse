from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar
from PyQt5.QtCore import Qt, QTimer
import numpy as np
import sys

class VolumeWindow(QWidget):
    def __init__(self, device_name=None):
        # Check if we can create a GUI window
        try:
            super().__init__(None)  # Explicitly set no parent
            
            # Initialize volume tracking
            self.last_volume = 0.0
            self.smoothing_factor = 0.4  # Increased for faster response while maintaining stability
            self.setAttribute(Qt.WA_DeleteOnClose)  # Ensure proper cleanup
            self.setWindowTitle("Microphone Volume")
            self.setFixedSize(300, 200)
            self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
            
            # Set window and widget attributes for proper rendering
            self.setAttribute(Qt.WA_StyledBackground, True)
            self.setAutoFillBackground(True)
            
            # Set window background color
            palette = self.palette()
            palette.setColor(self.backgroundRole(), Qt.white)
            palette.setColor(self.foregroundRole(), Qt.black)
            self.setPalette(palette)
            
            # Set stylesheet for the entire window and its widgets
            self.setStyleSheet("""
                VolumeWindow {
                    background-color: white;
                }
                QLabel {
                    color: black;
                    background-color: white;
                    padding: 2px;
                }
                QProgressBar {
                    border: 1px solid #cccccc;
                    background-color: #f0f0f0;
                    height: 15px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #2196f3;
                }
            """)
            
            # Create main layout
            layout = QVBoxLayout()
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(15)
            
            # Device name label with explicit background
            self.device_label = QLabel(device_name or "No device selected")
            self.device_label.setWordWrap(True)
            self.device_label.setAlignment(Qt.AlignCenter)
            self.device_label.setAutoFillBackground(True)
            palette = self.device_label.palette()
            palette.setColor(self.device_label.backgroundRole(), Qt.white)
            self.device_label.setPalette(palette)
            layout.addWidget(self.device_label)
            
            # Volume label with explicit background
            self.volume_label = QLabel("Volume Level")
            self.volume_label.setAlignment(Qt.AlignCenter)
            self.volume_label.setAutoFillBackground(True)
            palette = self.volume_label.palette()
            palette.setColor(self.volume_label.backgroundRole(), Qt.white)
            self.volume_label.setPalette(palette)
            font = self.volume_label.font()
            font.setBold(True)
            font.setPointSize(12)
            self.volume_label.setFont(font)
            layout.addWidget(self.volume_label)
            
            # Progress bar for volume
            self.volume_bar = QProgressBar()
            self.volume_bar.setMinimum(0)
            self.volume_bar.setMaximum(100)
            self.volume_bar.setValue(0)
            layout.addWidget(self.volume_bar)
            
            # Numeric volume level with explicit background
            self.level_label = QLabel("0 dB")
            self.level_label.setAlignment(Qt.AlignCenter)
            self.level_label.setAutoFillBackground(True)
            palette = self.level_label.palette()
            palette.setColor(self.level_label.backgroundRole(), Qt.white)
            self.level_label.setPalette(palette)
            font = self.level_label.font()
            font.setPointSize(11)
            self.level_label.setFont(font)
            layout.addWidget(self.level_label)
            
            # Set layout and show window
            self.setLayout(layout)
            self.show()
            
            # Force an initial paint
            self.repaint()
            QApplication.processEvents()
            
            # Initialize state
            self.running = True
            self.has_gui = True
            
            # Force initial update
            self.update()
            
            print("\nVolume meter window opened successfully")
            
        except Exception as e:
            print(f"\nWarning: Could not create GUI window ({e})")
            print("Volume meter will not be displayed")
            self.running = False
            self.has_gui = False
        
    def update_volume(self, volume):
        """Update volume display"""
        if not self.running or not self.has_gui:
            return
            
        try:
            # Apply smoothing to volume changes
            smoothed_volume = (volume * self.smoothing_factor) + (self.last_volume * (1 - self.smoothing_factor))
            self.last_volume = smoothed_volume
            
            # Convert volume (0-1) directly to progress bar value (0-100)
            normalized = min(100, max(0, smoothed_volume * 100))
            
            # Use invokeMethod to ensure updates happen in the Qt thread
            QTimer.singleShot(0, lambda: self._update_widgets(normalized))
        except Exception:
            pass  # Ignore errors if window is being destroyed
    
    def _update_widgets(self, normalized):
        """Update GUI widgets (called in Qt thread)"""
        if not self.running or not self.has_gui:
            return
        try:
            value = int(normalized)
            self.volume_bar.setValue(value)
            self.level_label.setText(f"{value}%")
            # Force an immediate update
            self.volume_bar.repaint()
            self.level_label.repaint()
        except Exception:
            pass  # Ignore errors if window is being destroyed
    
    def update(self):
        """Override update method"""
        if self.running and self.has_gui:
            try:
                # Update all widgets
                super().update()
                for widget in [self.device_label, self.volume_label, self.volume_bar, self.level_label]:
                    if widget:
                        widget.update()
            except Exception as e:
                print(f"Error updating widgets: {e}")
    
    def closeEvent(self, event):
        """Handle window close button"""
        self.running = False
        self.has_gui = False
        event.accept()
    
    def close(self):
        """Close the window"""
        if self.running and self.has_gui:
            self.running = False
            self.has_gui = False
            try:
                super().close()
            except Exception as e:
                print(f"Error closing window: {e}")
