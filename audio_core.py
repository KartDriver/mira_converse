"""
Core audio processing functionality with professional-grade audio processing,
noise floor estimation, and speech detection.
"""

import numpy as np
import time
from collections import deque
from scipy import signal
import sounddevice as sd
import platform
import warnings
import io
import soundfile as sf

class AudioCore:
    def __init__(self):
        # Professional audio time constants
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
        self.min_floor = -65.0
        self.max_floor = -20.0
        
        # Advanced floor estimation
        self.window_size = 150    # 3 seconds at 50Hz updates
        self.recent_levels = deque(maxlen=self.window_size)
        self.min_history = deque(maxlen=20)  # Longer history for stability
        
        # Speech detection with hysteresis
        self.speech_threshold_open = 2.5    # Open threshold above floor (dB)
        self.speech_threshold_close = 1.5   # Close threshold above floor (dB)
        self.is_speaking = False
        self.hold_counter = 0
        self.hold_samples = 15     # Hold samples at 50Hz update rate (0.3s at 50Hz)
        self.debug_last_state = False  # For state change logging
        
        # Pre-emphasis filter
        self.pre_emphasis = 0.97
        self.prev_sample = 0.0

        # Audio device configuration
        self.CHUNK = 2048  # Larger chunk size for better processing
        self.CHANNELS = 1
        self.DESIRED_RATE = 16000

    def init_audio_device(self):
        """Initialize audio device with proper configuration"""
        try:
            # Print available devices for debugging
            print("\nAvailable audio devices:")
            print(sd.query_devices())
            
            # Find a suitable input device based on system
            system = platform.system().lower()
            
            # On Linux, try to suppress common ALSA warnings
            if system == 'linux':
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="sounddevice")
            
            # Try to find built-in microphone based on system
            devices = sd.query_devices()
            working_device = None
            device_info = None
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    device_name = device['name'].lower()
                    
                    # On macOS, prefer the built-in microphone
                    if system == 'darwin' and 'macbook' in device_name and 'microphone' in device_name:
                        print("\nSelected MacBook's built-in microphone")
                        working_device = i
                        device_info = device
                        break
                        
                    # On Linux, prioritize AMD audio device for Lenovo laptops
                    elif system == 'linux':
                        # First preference: AMD audio device
                        if 'acp' in device_name:
                            print(f"\nSelected AMD audio device: {device['name']}")
                            working_device = i
                            device_info = device
                            break
                        # Second preference: Default device
                        try:
                            default_info = sd.query_default_device('input')
                            if default_info[0] == i:
                                print(f"\nSelected default ALSA device: {device['name']}")
                                working_device = i
                                device_info = device
                                break
                        except Exception:
                            pass
            
            # Fall back to default input device if no specific device found
            if working_device is None:
                try:
                    default_device = sd.query_default_device('input')[0]
                    device_info = sd.query_devices(default_device)
                    working_device = default_device
                    print(f"\nUsing default input device: {device_info['name']}")
                except Exception as e:
                    print(f"Error getting default device: {e}")
                    # Last resort: use first available input device
                    for i, device in enumerate(devices):
                        if device['max_input_channels'] > 0:
                            working_device = i
                            device_info = device
                            print(f"\nUsing first available input device: {device['name']}")
                            break
            
            if working_device is None or device_info is None:
                raise RuntimeError("No suitable input device found")
            
            # Configure stream parameters
            rate = int(device_info['default_samplerate'])
            needs_resampling = rate != self.DESIRED_RATE
            
            # Print detailed device info
            print("\nSelected device details:")
            print(f"  Name: {device_info['name']}")
            print(f"  Input channels: {device_info['max_input_channels']}")
            print(f"  Default samplerate: {rate}")
            print(f"  Low latency: {device_info['default_low_input_latency']}")
            print(f"  High latency: {device_info['default_high_input_latency']}")
            
            # Create input stream with optimal settings
            stream = sd.InputStream(
                device=working_device,
                channels=1,
                samplerate=rate,
                dtype=np.float32,
                blocksize=self.CHUNK,
                latency='low'  # Use low latency for better responsiveness
            )
            
            print("\nStarting stream...")
            stream.start()
            
            # Verify stream is working
            test_data = stream.read(self.CHUNK)[0]
            if np.all(test_data == 0):
                print("WARNING: Microphone input is all zeros!")
            elif np.max(np.abs(test_data)) < 0.001:
                print("WARNING: Very low microphone input levels!")
            else:
                level = 20 * np.log10(np.max(np.abs(test_data)) + 1e-10)
                rms = 20 * np.log10(np.sqrt(np.mean(test_data**2)) + 1e-10)
                print(f"\nMicrophone test successful:")
                print(f"  Peak level: {level:.1f} dB")
                print(f"  RMS level: {rms:.1f} dB")
                print(f"  Sample rate: {rate} Hz")
            
            return stream, device_info, rate, needs_resampling
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize audio: {str(e)}")

    def bytes_to_float32_audio(self, audio_data, sample_rate=None):
        """Convert bytes to float32 audio data"""
        # Convert bytes to int16 numpy array
        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        # Convert to float32 and normalize to [-1.0, 1.0]
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        # Use provided sample rate or default to DESIRED_RATE
        return audio_float32, (sample_rate if sample_rate is not None else self.DESIRED_RATE)

    def process_audio(self, audio_data):
        """Process audio with professional techniques"""
        # Check for empty or invalid input
        if len(audio_data) == 0:
            return {
                'audio': np.array([]),
                'is_speech': False,
                'db_level': self.rms_level,
                'noise_floor': self.noise_floor,
                'speech_ratio': 0,
                'zero_crossings': 0,
                'peak_level': self.peak_level
            }
            
        # Remove DC offset (with empty array check)
        if len(audio_data) > 0:
            dc_removed = audio_data - np.mean(audio_data)
        else:
            dc_removed = np.array([])
        
        # Apply pre-emphasis filter (with empty array check)
        emphasized = np.zeros_like(dc_removed)
        if len(dc_removed) > 0:
            emphasized[0] = dc_removed[0] - self.pre_emphasis * self.prev_sample
            if len(dc_removed) > 1:
                emphasized[1:] = dc_removed[1:] - self.pre_emphasis * dc_removed[:-1]
            self.prev_sample = dc_removed[-1]
        
        # Update levels with envelope following
        now = time.time()
        dt = now - self.last_update
        self.last_update = now
        
        rms = np.sqrt(np.mean(emphasized**2))
        peak = np.max(np.abs(emphasized))
        
        # Convert to dB
        db_rms = 20 * np.log10(max(rms, 1e-10))
        db_peak = 20 * np.log10(max(peak, 1e-10))
        
        # Professional envelope following
        if db_rms > self.rms_level:
            alpha = 1.0 - np.exp(-dt / self.rms_attack)
        else:
            alpha = 1.0 - np.exp(-dt / self.rms_release)
        self.rms_level = self.rms_level + (db_rms - self.rms_level) * alpha
        
        if db_peak > self.peak_level:
            alpha = 1.0 - np.exp(-dt / self.peak_attack)
        else:
            alpha = 1.0 - np.exp(-dt / self.peak_release)
        self.peak_level = self.peak_level + (db_peak - self.peak_level) * alpha
        
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

        # Spectral analysis for speech detection with dynamic window size
        if len(emphasized) > 0:
            # Use smaller window size for short segments
            nperseg = min(256, len(emphasized))
            # Ensure noverlap is less than nperseg
            noverlap = min(nperseg - 1, nperseg // 2)
            
            try:
                freqs, times, Sxx = signal.spectrogram(
                    emphasized, 
                    fs=16000,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    scaling='spectrum'
                )
                speech_mask = (freqs >= 100) & (freqs <= 3500)
                speech_energy = np.mean(Sxx[speech_mask, :]) if Sxx.size > 0 else 0
                total_energy = np.mean(Sxx) if Sxx.size > 0 else 0
                speech_ratio = speech_energy / total_energy if total_energy > 0 else 0
            except Exception as e:
                print(f"Spectrogram analysis failed: {e}")
                speech_ratio = 0
        else:
            speech_ratio = 0
        
        # Calculate zero-crossings
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(emphasized)))) / len(emphasized)
        
        # Speech detection with spectral analysis and hysteresis
        has_speech_character = speech_ratio > 1.02 and zero_crossings > 0.0002
        
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

        # Log state changes for debugging
        if self.is_speaking != self.debug_last_state:
            if self.is_speaking:
                print(f"\nSpeech START - Level: {self.rms_level:.1f}dB, Floor: {self.noise_floor:.1f}dB, Ratio: {speech_ratio:.3f}")
            else:
                print(f"\nSpeech END - Level: {self.rms_level:.1f}dB, Floor: {self.noise_floor:.1f}dB, Ratio: {speech_ratio:.3f}")
            self.debug_last_state = self.is_speaking

        return {
            'audio': emphasized,
            'is_speech': self.is_speaking,
            'db_level': self.rms_level,
            'noise_floor': self.noise_floor,
            'speech_ratio': speech_ratio,
            'zero_crossings': zero_crossings,
            'peak_level': self.peak_level
        }

    def calculate_volume(self, audio_data):
        """Calculate volume using professional audio metering"""
        if self.rms_level > self.noise_floor:
            # Professional audio compression curve
            db_above_floor = self.rms_level - self.noise_floor
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

    def get_thresholds(self):
        """Get current threshold and level values"""
        return {
            'noise_floor': self.noise_floor,
            'speech_threshold_open': self.noise_floor + self.speech_threshold_open,
            'speech_threshold_close': self.noise_floor + self.speech_threshold_close,
            'rms_level': self.rms_level,
            'peak_level': self.peak_level
        }
