"""
Core audio processing functionality with professional-grade audio processing,
noise floor estimation, and speech detection.
"""

import json
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
    @property
    def noise_floor(self):
        return self._noise_floor
            
    @noise_floor.setter
    def noise_floor(self, value):
        try:
            if value is not None:
                val = float(value)
                # Allow slightly lower values but clamp to safe range
                self._noise_floor = max(-150.0, min(val, -20.0))
        except Exception as e:
            print(f"Error setting noise floor: {e}")
            self._noise_floor = -96.0  # Safe default
            
    @property
    def min_floor(self):
        return self._min_floor
            
    @min_floor.setter
    def min_floor(self, value):
        try:
            if value is not None:
                val = float(value)
                # Allow slightly lower values but clamp to safe range
                self._min_floor = max(-150.0, min(val, -20.0))
        except Exception as e:
            print(f"Error setting min floor: {e}")
            self._min_floor = -96.0  # Safe default
            
    @property
    def max_floor(self):
        return self._max_floor
            
    @max_floor.setter
    def max_floor(self, value):
        try:
            if value is not None:
                val = float(value)
                # Allow slightly higher values but clamp to safe range
                self._max_floor = max(-100.0, min(val, 0.0))
        except Exception as e:
            print(f"Error setting max floor: {e}")
            self._max_floor = -36.0  # Safe default
            
    @property
    def rms_level(self):
        return self._rms_level
            
    @rms_level.setter
    def rms_level(self, value):
        try:
            if value is not None:
                val = float(value)
                # Allow slightly lower values but clamp to safe range
                self._rms_level = max(-150.0, min(val, -20.0))
        except Exception as e:
            print(f"Error setting RMS level: {e}")
            self._rms_level = -96.0  # Safe default
            
    @property
    def peak_level(self):
        return self._peak_level
            
    @peak_level.setter
    def peak_level(self, value):
        try:
            if value is not None:
                val = float(value)
                # Allow slightly lower values but clamp to safe range
                self._peak_level = max(-150.0, min(val, -20.0))
        except Exception as e:
            print(f"Error setting peak level: {e}")
            self._peak_level = -96.0  # Safe default
            
    @property
    def current_db_rms(self):
        return self._current_db_rms
            
    @current_db_rms.setter
    def current_db_rms(self, value):
        try:
            if value is not None:
                val = float(value)
                # Allow slightly lower values but clamp to safe range
                self._current_db_rms = max(-150.0, min(val, -20.0))
        except Exception as e:
            print(f"Error setting current RMS: {e}")
            self._current_db_rms = -96.0  # Safe default
            
    @property
    def current_db_peak(self):
        return self._current_db_peak
            
    @current_db_peak.setter
    def current_db_peak(self, value):
        try:
            if value is not None:
                val = float(value)
                # Allow slightly lower values but clamp to safe range
                self._current_db_peak = max(-150.0, min(val, -20.0))
        except Exception as e:
            print(f"Error setting current peak: {e}")
            self._current_db_peak = -96.0  # Safe default
            
    def __init__(self):
        # Load configuration
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Audio stream state
        self.stream = None
        self.rate = None
        self.needs_resampling = None
        
        # Voice profile storage
        self.voice_profile = None
        self.profile_timestamp = None
        self.profile_similarity_threshold = 0.3  # More permissive threshold for better trigger detection
        
        # Professional audio time constants from config
        self.peak_attack = config['audio_processing']['time_constants']['peak_attack']
        self.peak_release = config['audio_processing']['time_constants']['peak_release']
        self.rms_attack = config['audio_processing']['time_constants']['rms_attack']
        self.rms_release = config['audio_processing']['time_constants']['rms_release']
        
        # Level detection with safe defaults
        self._peak_level = -96.0
        self._rms_level = -96.0
        self._current_db_rms = -96.0
        self._current_db_peak = -96.0
        self.last_update = time.time()
        
        # Simple noise floor tracking with safe defaults
        self._noise_floor = -96.0  # Start with safe default
        self._min_floor = -96.0    # Start with safe default
        self._max_floor = -36.0    # Start with safe default (60dB range)
        
        # Calibration state
        self.calibration_samples = []
        self.last_calibration = 0
        self.CALIBRATION_INTERVAL = 5.0  # Seconds between recalibration checks
        self.CALIBRATION_WINDOW = 1.0    # Seconds of samples to use for calibration
        self.level_history = deque(maxlen=50)  # Short history for level stability check
        
        # Speech detection with hysteresis from config
        self.speech_threshold_open = config['speech_detection']['thresholds']['open']
        self.speech_threshold_close = config['speech_detection']['thresholds']['close']
        self.is_speaking = False
        self.hold_counter = 0
        self.hold_samples = config['speech_detection']['hold_samples']
        self.debug_last_state = False  # For state change logging
        
        # Pre-emphasis filter from config
        self.pre_emphasis = config['speech_detection']['pre_emphasis']
        self.prev_sample = 0.0

        # Audio device configuration from config
        self.CHUNK = config['audio_processing']['chunk_size']
        self.CHANNELS = 1
        self.DESIRED_RATE = config['audio_processing']['desired_rate']

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
            
            # Verify stream is working and find minimum level over half a second
            print("\nMeasuring minimum ambient level...")
            chunks_for_half_second = int(rate / self.CHUNK / 2)  # Number of chunks in 0.5 seconds
            min_rms = float('inf')
            
            # Take multiple samples and find the minimum RMS
            retries = 3  # Allow a few retries to get valid samples
            while retries > 0 and min_rms == float('inf'):
                for _ in range(chunks_for_half_second):
                    chunk = stream.read(self.CHUNK)[0]
                    if not np.all(chunk == 0):  # Skip silent chunks
                        rms = 20 * np.log10(np.sqrt(np.mean(chunk**2)) + 1e-10)
                        min_rms = min(min_rms, rms)
                if min_rms == float('inf'):
                    retries -= 1
                    time.sleep(0.1)  # Short delay before retry
            
            if min_rms == float('inf'):
                print("\nWARNING: Could not get valid audio samples after multiple attempts!")
                return None, None, None, None  # Return None to indicate calibration failure
            
            # Set initial levels - never allow None values after initialization
            self.noise_floor = float(min_rms)
            self.min_floor = float(min_rms)  # Keep minimum at noise floor
            self.max_floor = float(min_rms + 60)  # Allow 60dB dynamic range above noise floor
            self.rms_level = float(min_rms)  # Initialize RMS to noise floor
            self.peak_level = float(min_rms)  # Initialize peak to noise floor
                
            print(f"\nMicrophone calibration successful:")
            print(f"  Minimum RMS level: {min_rms:.1f} dB")
            print(f"  Initial noise floor: {self.noise_floor:.1f} dB")
            print(f"  Floor range: {self.min_floor:.1f} to {self.max_floor:.1f} dB")
            print(f"  Sample rate: {rate} Hz")
            
            # Store stream info
            self.stream = stream
            self.rate = rate
            self.needs_resampling = needs_resampling
            
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

    def create_voice_profile(self, audio_data):
        """
        Create a voice profile from an audio segment.
        Returns a dictionary containing spectral characteristics.
        """
        # Ensure audio data is valid
        if len(audio_data) == 0:
            return None
            
        # Remove DC offset and apply pre-emphasis
        dc_removed = audio_data - np.mean(audio_data)
        emphasized = np.zeros_like(dc_removed)
        emphasized[0] = dc_removed[0]
        emphasized[1:] = dc_removed[1:] - self.pre_emphasis * dc_removed[:-1]
        
        # Calculate spectral characteristics
        nperseg = min(256, len(emphasized))
        noverlap = min(nperseg - 1, nperseg // 2)
        
        try:
            freqs, _, Sxx = signal.spectrogram(
                emphasized,
                fs=16000,
                nperseg=nperseg,
                noverlap=noverlap,
                scaling='spectrum'
            )
            
            # Focus on speech frequency bands
            speech_mask = (freqs >= 100) & (freqs <= 3500)
            speech_freqs = freqs[speech_mask]
            speech_power = np.mean(Sxx[speech_mask, :], axis=1)
            
            # Calculate additional voice characteristics
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(emphasized)))) / len(emphasized)
            rms = np.sqrt(np.mean(emphasized**2))
            peak = np.max(np.abs(emphasized))
            
            # Create profile
            profile = {
                'spectral_signature': speech_power,
                'frequency_bands': speech_freqs,
                'zero_crossing_rate': zero_crossings,
                'rms_level': 20 * np.log10(max(rms, 1e-10)),
                'peak_level': 20 * np.log10(max(peak, 1e-10)),
                'timestamp': time.time()
            }
            
            return profile
            
        except Exception as e:
            print(f"Error creating voice profile: {e}")
            return None

    def compare_voice_profile(self, audio_data, profile):
        """
        Compare audio segment with stored voice profile.
        Returns similarity score between 0 and 1.
        """
        if profile is None or len(audio_data) == 0:
            return 0.0
            
        try:
            # Create temporary profile for comparison
            current_profile = self.create_voice_profile(audio_data)
            if current_profile is None:
                return 0.0
                
            # Compare spectral signatures
            ref_spectrum = profile['spectral_signature']
            cur_spectrum = current_profile['spectral_signature']
            
            # Ensure spectra are the same length
            min_len = min(len(ref_spectrum), len(cur_spectrum))
            ref_spectrum = ref_spectrum[:min_len]
            cur_spectrum = cur_spectrum[:min_len]
            
            # Calculate normalized cross-correlation
            spectrum_similarity = np.corrcoef(ref_spectrum, cur_spectrum)[0, 1]
            if np.isnan(spectrum_similarity):
                spectrum_similarity = 0.0
            
            # Compare other characteristics with tolerance
            zcr_similarity = 1.0 - min(1.0, abs(profile['zero_crossing_rate'] - 
                                               current_profile['zero_crossing_rate']) / 0.01)
            level_similarity = 1.0 - min(1.0, abs(profile['rms_level'] - 
                                                 current_profile['rms_level']) / 20.0)
            
            # Weighted combination of similarities (more weight on basic characteristics)
            total_similarity = (0.2 * max(0, spectrum_similarity) + 
                              0.4 * zcr_similarity +      # More weight on pitch/timing
                              0.4 * level_similarity)     # More weight on volume patterns
            
            return max(0.0, min(1.0, total_similarity))
            
        except Exception as e:
            print(f"Error comparing voice profiles: {e}")
            return 0.0

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
        
        try:
            # Convert to dB with safety checks
            current_db_rms = float(20 * np.log10(max(rms, 1e-10)))
            current_db_peak = float(20 * np.log10(max(peak, 1e-10)))
            
            # Store current values through property setters
            self.current_db_rms = current_db_rms
            self.current_db_peak = current_db_peak
            
            # Professional envelope following with safety
            if self.rms_level is None:
                self.rms_level = current_db_rms
            else:
                if current_db_rms > self.rms_level:
                    alpha = 1.0 - np.exp(-dt / self.rms_attack)
                else:
                    alpha = 1.0 - np.exp(-dt / self.rms_release)
                self.rms_level = float(self.rms_level + (current_db_rms - self.rms_level) * alpha)
            
            if self.peak_level is None:
                self.peak_level = current_db_peak
            else:
                if current_db_peak > self.peak_level:
                    alpha = 1.0 - np.exp(-dt / self.peak_attack)
                else:
                    alpha = 1.0 - np.exp(-dt / self.peak_release)
                self.peak_level = float(self.peak_level + (current_db_peak - self.peak_level) * alpha)
        except Exception as e:
            print(f"Error updating levels: {e}")
            # Use safe defaults if calculation fails
            self.current_db_rms = -96.0
            self.current_db_peak = -96.0
            self.rms_level = -96.0
            self.peak_level = -96.0
        
        # Track audio levels for noise floor calibration
        self.level_history.append(self.rms_level)
        self.calibration_samples.append(self.rms_level)
        
        # Periodic recalibration (noise floor should never be None after init)
        if now - self.last_calibration > self.CALIBRATION_INTERVAL:
            # Only recalibrate if levels have been stable (no speech)
            level_std = np.std(self.level_history)
            if level_std < 3.0:  # If levels haven't varied much
                # Find minimum level in recent history
                min_level = min(self.level_history)
                # Only adjust if we found a new minimum
                if min_level < self.noise_floor:
                    self.noise_floor = min_level
                    self.min_floor = self.noise_floor  # Keep minimum at noise floor
                    self.max_floor = self.noise_floor + 60  # Maintain 60dB range
                    print(f"\nRecalibrated noise floor to: {self.noise_floor:.1f} dB")
                self.last_calibration = now

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
        
        # Use configured thresholds relative to noise floor
        open_threshold = self.speech_threshold_open  # dB above noise floor to open gate
        close_threshold = self.speech_threshold_close  # dB above noise floor to close gate
        
        # Calculate level above floor for debugging with safety checks
        try:
            # Ensure we have valid values
            if self.rms_level is None:
                self.rms_level = float(self.current_db_rms)
            if self.noise_floor is None:
                # If we lost calibration, reinitialize all levels
                self.noise_floor = float(self.current_db_rms)
                self.min_floor = float(self.current_db_rms)
                self.max_floor = float(self.current_db_rms + 60)
                print(f"\nWarning: Reinitializing noise floor to {self.noise_floor:.1f} dB")
                
            # Calculate level with explicit float conversion
            level_above_floor = float(self.rms_level) - float(self.noise_floor)
        except Exception as e:
            print(f"\nError calculating level above floor: {e}")
            level_above_floor = 0.0
        
        if not self.is_speaking:
            # Check if should open gate
            is_speech = (level_above_floor > open_threshold and has_speech_character)
            if is_speech:
                # print(f"\n[DEBUG] Speech gate opening:")
                # print(f"  Level: {self.rms_level:.1f} dB")
                # print(f"  Floor: {self.noise_floor:.1f} dB")
                # print(f"  Above floor: {level_above_floor:.1f} dB")
                # print(f"  Speech ratio: {speech_ratio:.3f}")
                # print(f"  Zero crossings: {zero_crossings:.4f}")
                self.is_speaking = True
                self.hold_counter = 50  # Increased hold samples for better phrase detection
        else:
            # Check if should close gate
            if level_above_floor < close_threshold:
                if self.hold_counter > 0:
                    self.hold_counter -= 1
                    is_speech = True
                    # if self.hold_counter % 10 == 0:  # Print debug every 10 samples
                    #     print(f"\n[DEBUG] Hold counter: {self.hold_counter}")
                    #     print(f"  Level: {self.rms_level:.1f} dB")
                    #     print(f"  Above floor: {level_above_floor:.1f} dB")
                else:
                    # print(f"\n[DEBUG] Speech gate closing:")
                    # print(f"  Level: {self.rms_level:.1f} dB")
                    # print(f"  Floor: {self.noise_floor:.1f} dB")
                    # print(f"  Above floor: {level_above_floor:.1f} dB")
                    # print(f"  Speech ratio: {speech_ratio:.3f}")
                    # print(f"  Zero crossings: {zero_crossings:.4f}")
                    self.is_speaking = False
                    is_speech = False
            else:
                self.hold_counter = 50  # Reset hold counter while still above threshold
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
        # Safety check for valid levels
        if self.rms_level is None or self.noise_floor is None or self.max_floor is None:
            return 0.0
            
        if self.rms_level > self.noise_floor:
            # Professional audio compression curve
            db_above_floor = float(self.rms_level - self.noise_floor)  # Ensure float arithmetic
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
        try:
            # Return safe values if any level is invalid
            if any(x is None for x in [self.noise_floor, self.rms_level, self.peak_level]):
                return {
                    'noise_floor': -96.0,
                    'speech_threshold_open': -66.0,  # -96 + 30
                    'speech_threshold_close': -71.0, # -96 + 25
                    'rms_level': -96.0,
                    'peak_level': -96.0
                }
            
            # Calculate thresholds with explicit float conversions
            noise_floor = float(self.noise_floor)
            return {
                'noise_floor': noise_floor,
                'speech_threshold_open': float(noise_floor + self.speech_threshold_open),
                'speech_threshold_close': float(noise_floor + self.speech_threshold_close),
                'rms_level': float(self.rms_level),
                'peak_level': float(self.peak_level)
            }
        except Exception as e:
            print(f"Error getting thresholds: {e}")
            # Return safe defaults on error
            return {
                'noise_floor': -96.0,
                'speech_threshold_open': -66.0,
                'speech_threshold_close': -71.0,
                'rms_level': -96.0,
                'peak_level': -96.0
            }

    def close(self):
        """Clean up audio resources"""
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            except Exception as e:
                print(f"Error closing audio stream: {e}")
