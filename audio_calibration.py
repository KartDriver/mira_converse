"""
Professional audio processing system with advanced noise floor estimation and speech detection.
Implements techniques from professional audio gates and speech processing systems.
"""

import numpy as np
from collections import deque
import time
from scipy import signal

class AudioCalibration:
    def __init__(self):
        # Professional audio time constants (in seconds)
        self.peak_attack = 0.001   # 1ms peak attack
        self.peak_release = 0.100  # 100ms peak release
        self.rms_attack = 0.030    # 30ms RMS attack
        self.rms_release = 0.500   # 500ms RMS release
        
        # Level detection
        self.peak_level = -96.0
        self.rms_level = -96.0
        self.last_update = time.time()
        
        # Noise floor tracking
        self.noise_floor = -50.0
        self.min_floor = -65.0    # Extended range for better sensitivity
        self.max_floor = -20.0
        
        # Advanced floor estimation
        self.window_size = 150    # 3 seconds at 50Hz updates
        self.recent_levels = deque(maxlen=self.window_size)
        self.min_history = deque(maxlen=20)  # Longer history for stability
        
        # Speech detection with hysteresis
        self.speech_threshold_open = 3.0    # Open threshold above floor (dB)
        self.speech_threshold_close = 2.0   # Close threshold above floor (dB)
        self.is_gate_open = False
        self.hold_counter = 0
        self.hold_samples = 10     # Hold samples at 50Hz update rate
        
        # Pre-emphasis filter coefficients (first-order high-pass at ~100Hz)
        self.pre_emphasis = 0.95   # Slightly reduced for less aggressive filtering
        self.prev_sample = 0.0
        
        # Initialize running statistics
        self.update_stats()
    
    def update_stats(self):
        """Update level detection and noise floor using professional audio techniques"""
        if not self.recent_levels:
            return
            
        now = time.time()
        dt = now - self.last_update
        self.last_update = now
        
        # Get current minimum level estimate (15th percentile for better sensitivity)
        current_min = np.percentile(self.recent_levels, 15)
        self.min_history.append(current_min)
        
        # Use weighted average of recent minimums for base floor
        weights = np.exp(-np.arange(len(self.min_history)) / 10)
        base_floor = np.average(self.min_history, weights=weights)
        
        # Professional envelope following for noise floor
        if base_floor < self.noise_floor:
            # Fast attack with smoothing
            alpha_attack = 1.0 - np.exp(-dt / 0.100)  # 100ms attack
            self.noise_floor = max(
                self.min_floor,
                self.noise_floor + (base_floor - self.noise_floor) * alpha_attack
            )
        else:
            # Slow release with adaptive time constant
            level_diff = base_floor - self.noise_floor
            release_time = np.interp(level_diff, [0, 20], [2.0, 5.0])
            alpha_release = 1.0 - np.exp(-dt / release_time)
            self.noise_floor = min(
                self.max_floor,
                self.noise_floor + (base_floor - self.noise_floor) * alpha_release
            )
    
    def process_audio_chunk(self, audio_chunk):
        """Process audio chunk with professional techniques"""
        # Remove DC offset
        dc_removed = audio_chunk - np.mean(audio_chunk)
        
        # Apply pre-emphasis filter
        emphasized = np.zeros_like(dc_removed)
        emphasized[0] = dc_removed[0] - self.pre_emphasis * self.prev_sample
        emphasized[1:] = dc_removed[1:] - self.pre_emphasis * dc_removed[:-1]
        self.prev_sample = dc_removed[-1]
        
        # Calculate RMS and peak levels
        rms = np.sqrt(np.mean(emphasized**2))
        peak = np.max(np.abs(emphasized))
        
        # Convert to dB with proper floor
        db_rms = 20 * np.log10(max(rms, 1e-10))
        db_peak = 20 * np.log10(max(peak, 1e-10))
        
        # Professional envelope following for levels
        dt = time.time() - self.last_update
        
        # Update RMS level
        if db_rms > self.rms_level:
            alpha = 1.0 - np.exp(-dt / self.rms_attack)
        else:
            alpha = 1.0 - np.exp(-dt / self.rms_release)
        self.rms_level = self.rms_level + (db_rms - self.rms_level) * alpha
        
        # Update peak level
        if db_peak > self.peak_level:
            alpha = 1.0 - np.exp(-dt / self.peak_attack)
        else:
            alpha = 1.0 - np.exp(-dt / self.peak_release)
        self.peak_level = self.peak_level + (db_peak - self.peak_level) * alpha
        
        return self.rms_level, emphasized
    
    def add_sample(self, audio_chunk):
        """Process audio chunk and update noise floor estimation"""
        rms_level, emphasized = self.process_audio_chunk(audio_chunk)
        self.recent_levels.append(rms_level)
        self.update_stats()
        
        # Calculate zero-crossings on DC-removed audio
        dc_removed = audio_chunk - np.mean(audio_chunk)
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(dc_removed)))) / len(dc_removed)
        
        # Determine if this is speech
        is_speech = self.is_speech(rms_level, zero_crossings)
        
        return is_speech, audio_chunk  # Return original audio for buffer
    
    def calculate_volume(self, audio_chunk):
        """Calculate volume using professional audio metering"""
        rms_level, _ = self.process_audio_chunk(audio_chunk)
        if rms_level > self.noise_floor:
            # Professional audio compression curve
            db_above_floor = rms_level - self.noise_floor
            ratio = 0.8  # Subtle compression ratio
            knee = 6.0   # Soft knee width in dB
            
            # Soft knee compression
            if db_above_floor < -knee/2:
                gain = db_above_floor
            elif db_above_floor > knee/2:
                gain = -knee/2 + (db_above_floor - (-knee/2)) / ratio
            else:
                # Smooth transition in knee region
                gain = db_above_floor + ((1/ratio - 1) * 
                       (db_above_floor + knee/2)**2 / (2*knee))
            
            # Convert to linear scale with proper normalization
            volume = np.power(10, gain/20) / np.power(10, (self.max_floor - self.noise_floor)/20)
            return max(0.05, min(1.0, volume))
        return 0.0
    
    def is_speech(self, rms_level, zero_crossings):
        """Professional speech detection with hysteresis"""
        # Basic speech detection with hysteresis
        if not self.is_gate_open:
            # Check if should open gate
            if (rms_level > self.noise_floor + self.speech_threshold_open and 
                zero_crossings > 0.0003):  # Reduced threshold
                self.is_gate_open = True
                self.hold_counter = self.hold_samples
        else:
            # Check if should close gate
            if rms_level < self.noise_floor + self.speech_threshold_close:
                if self.hold_counter > 0:
                    self.hold_counter -= 1
                else:
                    self.is_gate_open = False
            else:
                self.hold_counter = self.hold_samples
        
        return self.is_gate_open
    
    def get_thresholds(self):
        """Get current threshold and level values"""
        return {
            'noise_floor': self.noise_floor,
            'speech_threshold_open': self.noise_floor + self.speech_threshold_open,
            'speech_threshold_close': self.noise_floor + self.speech_threshold_close,
            'rms_level': self.rms_level,
            'peak_level': self.peak_level
        }
