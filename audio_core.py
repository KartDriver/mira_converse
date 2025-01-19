"""
Core audio processing functionality with professional-grade audio processing,
noise floor estimation, and speech detection.
"""

import numpy as np
import time
from collections import deque
from scipy import signal
import pyaudio
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
        self.CHUNK = 2048
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.DESIRED_RATE = 16000

    def init_audio_device(self):
        """Initialize audio device with proper configuration"""
        p = pyaudio.PyAudio()
        try:
            # Print available devices for debugging
            print("\nAvailable audio devices:")
            for i in range(p.get_device_count()):
                try:
                    dev_info = p.get_device_info_by_index(i)
                    print(f"Device {i}: {dev_info['name']} (inputs: {dev_info['maxInputChannels']})")
                except OSError:
                    print(f"Device {i}: <error accessing device>")

            # Find a suitable input device
            device_info = self._find_input_device(p)
            if device_info['maxInputChannels'] > 0:
                print(f"\nSelected audio device: {device_info['name']} (index: {device_info['index']})")
                
                # Check if device supports 16kHz directly
                rate = self.DESIRED_RATE
                needs_resampling = not self._is_rate_supported(p, device_info, self.DESIRED_RATE)
                
                if needs_resampling:
                    # Find the closest supported rate above 16kHz
                    supported_rates = [r for r in [16000, 22050, 32000, 44100, 48000] 
                                     if self._is_rate_supported(p, device_info, r)]
                    if not supported_rates:
                        raise ValueError("No suitable sample rates supported by the device")
                    rate = min(supported_rates)
                    print(f"\nDevice doesn't support 16kHz directly. Using {rate}Hz with resampling.")
                else:
                    print("\nDevice supports 16kHz recording directly.")
                
                return p, device_info, rate, needs_resampling
            else:
                raise ValueError("Selected device has no input channels")
        except Exception as e:
            p.terminate()
            raise RuntimeError(f"Failed to initialize audio: {str(e)}")

    def _is_rate_supported(self, p, device_info, rate):
        """Check if the given sample rate is supported by the device"""
        try:
            return p.is_format_supported(
                rate,
                input_device=device_info['index'],
                input_channels=self.CHANNELS,
                input_format=self.FORMAT
            )
        except ValueError:
            return False

    def _find_input_device(self, p):
        """Find a suitable audio input device with system-specific preferences"""
        system = platform.system().lower()
        
        # On Linux, try to suppress common ALSA warnings
        if system == 'linux':
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="sounddevice")

        # First try to find built-in microphone based on system
        for i in range(p.get_device_count()):
            try:
                device_info = p.get_device_info_by_index(i)
                device_name = device_info.get('name', '').lower()
                
                if device_info['maxInputChannels'] > 0:
                    # On macOS, prefer the built-in microphone
                    if system == 'darwin' and 'macbook' in device_name and 'microphone' in device_name:
                        print("\nSelected MacBook's built-in microphone")
                        return device_info
                        
                    # On Linux, prioritize AMD audio device for Lenovo laptops
                    elif system == 'linux':
                        # First preference: AMD audio device
                        if 'acp' in device_name:
                            print(f"\nSelected AMD audio device: {device_info['name']}")
                            return device_info
                        # Second preference: Default device
                        try:
                            default_info = p.get_default_input_device_info()
                            if default_info['index'] == device_info['index']:
                                print(f"\nSelected default ALSA device: {device_info['name']}")
                                return device_info
                        except OSError:
                            pass
            except OSError:
                continue

        # Try system default input device
        try:
            default_device_info = p.get_default_input_device_info()
            if default_device_info['maxInputChannels'] > 0:
                print("\nSelected system default input device")
                return default_device_info
        except OSError:
            pass

        # Fall back to first available input device
        for i in range(p.get_device_count()):
            try:
                device_info = p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    print("\nSelected first available input device")
                    return device_info
            except OSError:
                continue
        
        raise RuntimeError("No suitable input device found")

    def bytes_to_float32_audio(self, raw_bytes):
        """Convert raw bytes to float32 audio data"""
        audio_buffer = io.BytesIO(raw_bytes)
        audio_data, sample_rate = sf.read(audio_buffer, dtype='float32', 
                                        format='RAW', subtype='PCM_16', 
                                        samplerate=16000, channels=1)
        return audio_data, sample_rate

    def process_audio(self, audio_data):
        """Process audio with professional techniques"""
        # Remove DC offset
        dc_removed = audio_data - np.mean(audio_data)
        
        # Apply pre-emphasis filter
        emphasized = np.zeros_like(dc_removed)
        emphasized[0] = dc_removed[0] - self.pre_emphasis * self.prev_sample
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

        # Spectral analysis for speech detection
        freqs, times, Sxx = signal.spectrogram(emphasized, fs=16000, 
                                             nperseg=256, noverlap=128)
        speech_mask = (freqs >= 100) & (freqs <= 3500)
        speech_energy = np.mean(Sxx[speech_mask, :])
        total_energy = np.mean(Sxx)
        speech_ratio = speech_energy / total_energy if total_energy > 0 else 0
        
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
