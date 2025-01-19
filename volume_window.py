from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar
from PyQt5.QtCore import Qt, QTimer
import numpy as np
import sys
from collections import deque
import time
from scipy import signal

class VolumeWindow(QWidget):
    def __init__(self, device_name=None):
        # Check if we can create a GUI window
        try:
            super().__init__(None)  # Explicitly set no parent
            
            # Initialize audio processing (matching server logic)
            self.peak_level = -96.0
            self.rms_level = -96.0
            self.last_update = time.time()
            self.noise_floor = -50.0
            self.min_floor = -65.0
            self.max_floor = -20.0
            self.window_size = 150
            self.recent_levels = deque(maxlen=self.window_size)
            self.min_history = deque(maxlen=20)
            self.speech_threshold_open = 3.0
            self.speech_threshold_close = 2.0
            self.is_speaking = False
            self.hold_counter = 0
            self.hold_samples = 10
            self.pre_emphasis = 0.97
            self.prev_sample = 0.0
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
        
    def process_audio(self, audio_data):
        """Process audio using server's speech detection logic"""
        if not self.running or not self.has_gui:
            return
            
        try:
            # Remove DC offset
            dc_removed = audio_data - np.mean(audio_data)
            
            # Apply pre-emphasis filter (matching server)
            emphasized = np.zeros_like(dc_removed)
            emphasized[0] = dc_removed[0] - self.pre_emphasis * self.prev_sample
            emphasized[1:] = dc_removed[1:] - self.pre_emphasis * dc_removed[:-1]
            self.prev_sample = dc_removed[-1]
            
            # Update levels with envelope following
            now = time.time()
            dt = now - self.last_update
            self.last_update = now
            
            rms = np.sqrt(np.mean(emphasized**2))
            db_rms = 20 * np.log10(max(rms, 1e-10))
            
            # Update RMS level
            if db_rms > self.rms_level:
                alpha = 1.0 - np.exp(-dt / 0.030)  # 30ms attack
            else:
                alpha = 1.0 - np.exp(-dt / 0.500)  # 500ms release
            self.rms_level = self.rms_level + (db_rms - self.rms_level) * alpha
            
            # Update noise floor tracking
            self.recent_levels.append(self.rms_level)
            
            if len(self.recent_levels) > 0:
                current_min = np.percentile(self.recent_levels, 15)
                self.min_history.append(current_min)
                
                weights = np.exp(-np.arange(len(self.min_history)) / 10)
                base_floor = np.average(self.min_history, weights=weights)
                
                if base_floor < self.noise_floor:
                    alpha_attack = 1.0 - np.exp(-dt / 0.100)
                    self.noise_floor = max(
                        self.min_floor,
                        self.noise_floor + (base_floor - self.noise_floor) * alpha_attack
                    )
                else:
                    level_diff = base_floor - self.noise_floor
                    release_time = np.interp(level_diff, [0, 20], [2.0, 5.0])
                    alpha_release = 1.0 - np.exp(-dt / release_time)
                    self.noise_floor = min(
                        self.max_floor,
                        self.noise_floor + (base_floor - self.noise_floor) * alpha_release
                    )
            
            # Spectral analysis for speech detection (matching server)
            freqs, times, Sxx = signal.spectrogram(emphasized, fs=16000, 
                                                 nperseg=256, noverlap=128)
            speech_mask = (freqs >= 100) & (freqs <= 3500)
            speech_energy = np.mean(Sxx[speech_mask, :])
            total_energy = np.mean(Sxx)
            speech_ratio = speech_energy / total_energy if total_energy > 0 else 0
            
            # Calculate zero-crossings
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(emphasized)))) / len(emphasized)
            
            # Speech detection with spectral analysis and hysteresis
            has_speech_character = speech_ratio > 1.03 and zero_crossings > 0.0003
            
            if not self.is_speaking:
                # Check if should open gate
                is_speech = (self.rms_level > self.noise_floor + self.speech_threshold_open and 
                            has_speech_character)
                if is_speech:
                    self.is_speaking = True
                    self.hold_counter = self.hold_samples
            else:
                # Check if should close gate
                if self.rms_level < self.noise_floor + self.speech_threshold_close:
                    if self.hold_counter > 0:
                        self.hold_counter -= 1
                        is_speech = True
                    else:
                        self.is_speaking = False
                        is_speech = False
                else:
                    self.hold_counter = self.hold_samples
                    is_speech = True
            
            # Update volume display based on speech detection
            volume = 1.0 if is_speech else 0.0
            QTimer.singleShot(0, lambda: self._update_widgets(volume * 100))
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
