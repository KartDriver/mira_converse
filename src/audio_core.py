"""
Core audio processing functionality with robust noise floor calibration,
professional audio envelope following, and improved speech detection.

Key Points:
1. We store the config in 'self.config' so we can reference it in process_audio.
2. We read 'end_silence_duration' from config['speech_detection']['end_silence_duration'].
3. We read open/close thresholds from config['speech_detection']['thresholds']['open'/'close'].
4. Speech ends only after detecting the required duration of silence.
"""

import json
import numpy as np
import time
from collections import deque
from scipy import signal
import sounddevice as sd
import platform
import warnings

class AudioCore:
    @property
    def noise_floor(self):
        return self._noise_floor

    @noise_floor.setter
    def noise_floor(self, value):
        try:
            if value is not None:
                val = float(value)
                # Hard clamp to avoid absurd values
                self._noise_floor = max(-150.0, min(val, 20.0))
            else:
                self._noise_floor = -96.0
        except Exception as e:
            print(f"Error setting noise floor: {e}")
            self._noise_floor = -96.0

    @property
    def min_floor(self):
        return self._min_floor

    @min_floor.setter
    def min_floor(self, value):
        try:
            if value is not None:
                val = float(value)
                self._min_floor = max(-150.0, min(val, 20.0))
            else:
                self._min_floor = -96.0
        except Exception as e:
            print(f"Error setting min floor: {e}")
            self._min_floor = -96.0

    @property
    def max_floor(self):
        return self._max_floor

    @max_floor.setter
    def max_floor(self, value):
        try:
            if value is not None:
                val = float(value)
                self._max_floor = max(-150.0, min(val, 60.0))
            else:
                self._max_floor = -36.0
        except Exception as e:
            print(f"Error setting max floor: {e}")
            self._max_floor = -36.0

    @property
    def rms_level(self):
        return self._rms_level

    @rms_level.setter
    def rms_level(self, value):
        try:
            if value is not None:
                val = float(value)
                self._rms_level = max(-150.0, min(val, 20.0))
            else:
                self._rms_level = -96.0
        except Exception as e:
            print(f"Error setting RMS level: {e}")
            self._rms_level = -96.0

    @property
    def peak_level(self):
        return self._peak_level

    @peak_level.setter
    def peak_level(self, value):
        try:
            if value is not None:
                val = float(value)
                self._peak_level = max(-150.0, min(val, 20.0))
            else:
                self._peak_level = -96.0
        except Exception as e:
            print(f"Error setting peak level: {e}")
            self._peak_level = -96.0

    @property
    def current_db_rms(self):
        return self._current_db_rms

    @current_db_rms.setter
    def current_db_rms(self, value):
        try:
            if value is not None:
                val = float(value)
                self._current_db_rms = max(-150.0, min(val, 20.0))
            else:
                self._current_db_rms = -96.0
        except Exception as e:
            print(f"Error setting current RMS: {e}")
            self._current_db_rms = -96.0

    @property
    def current_db_peak(self):
        return self._current_db_peak

    @current_db_peak.setter
    def current_db_peak(self, value):
        try:
            if value is not None:
                val = float(value)
                self._current_db_peak = max(-150.0, min(val, 20.0))
            else:
                self._current_db_peak = -96.0
        except Exception as e:
            print(f"Error setting current peak: {e}")
            self._current_db_peak = -96.0

    def __init__(self):
        # Load config into self.config so we can reference it anywhere
        with open('config.json', 'r') as f:
            self.config = json.load(f)

        # Basic audio setup
        self.stream = None
        self.rate = None
        self.needs_resampling = None

        # Extract time constants from config
        self.peak_attack   = self.config['audio_processing']['time_constants']['peak_attack']
        self.peak_release  = self.config['audio_processing']['time_constants']['peak_release']
        self.rms_attack    = self.config['audio_processing']['time_constants']['rms_attack']
        self.rms_release   = self.config['audio_processing']['time_constants']['rms_release']

        # Initialize smoothed levels
        self._peak_level = -96.0
        self._rms_level = -96.0
        self._current_db_rms = -96.0
        self._current_db_peak = -96.0
        self.last_update = time.time()

        # Noise floor init
        self._noise_floor = -96.0
        self._min_floor   = -96.0
        self._max_floor   = -36.0

        # Additional calibration tracking
        self.noise_floor_ema = -96.0
        self.ema_alpha = 0.01
        self.calibration_samples = []
        self.last_calibration = 0
        self.CALIBRATION_INTERVAL = 30.0
        self.CALIBRATION_WINDOW   = 2.0
        self.level_history = deque(maxlen=50)

        # Read gating thresholds from config
        thresholds = self.config['speech_detection']['thresholds']
        self.speech_threshold_open  = thresholds['open']   # e.g. 12 dB in config
        self.speech_threshold_close = thresholds['close']  # e.g. 10 dB in config

        self.is_speaking = False
        self.hold_counter = 0
        self.debug_last_state = False

        # We'll no longer rely solely on consecutive frames, but keep them if needed
        self.consecutive_silence_needed = self.config['speech_detection']['hold_samples']
        self.current_silence_frames = 0

        # Ratio gating from config or defaults
        self.open_ratio     = 1.3
        self.close_ratio    = 1.1
        self.ratio_override = 2.0

        # Pre-emphasis from config
        self.pre_emphasis = self.config['speech_detection']['pre_emphasis']
        self.prev_sample = 0.0

        # For client usage
        audio_cfg = self.config['audio_processing']
        self.CHUNK        = audio_cfg['chunk_size']
        self.CHANNELS     = 1
        self.DESIRED_RATE = audio_cfg['desired_rate']

        self.debug_counter = 0

    def init_audio_device(self):
        """
        Initialize audio device and perform a robust single-buffer calibration
        by recording 4 seconds of audio continuously with sd.rec().
        Discards the top 10% loudest frames and computes a robust low-percentile RMS as the noise floor.
        Then opens a continuous InputStream for real-time capture.
        """
        try:
            print("\nListing audio devices:")
            print(sd.query_devices())

            system = platform.system().lower()
            if system == 'linux':
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="sounddevice")

            devices = sd.query_devices()
            working_device = None
            device_info = None

            # Device selection logic
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    device_name = device['name'].lower()

                    if system == 'darwin' and 'microphone' in device_name:
                        working_device = i
                        device_info = device
                        break
                    elif system == 'linux' and 'acp' in device_name:
                        working_device = i
                        device_info = device
                        break

                    # fallback: check if it's the default input device
                    try:
                        default_info = sd.query_default_device('input')
                        if default_info[0] == i:
                            working_device = i
                            device_info = device
                            break
                    except:
                        pass

            if working_device is None:
                # final fallback
                try:
                    default_idx = sd.query_default_device('input')[0]
                    device_info = sd.query_devices(default_idx)
                    working_device = default_idx
                    print(f"\nUsing default input device: {device_info['name']}")
                except Exception as e:
                    print(f"No default input device found: {e}")
                    for i, dev in enumerate(devices):
                        if dev['max_input_channels'] > 0:
                            working_device = i
                            device_info = dev
                            print(f"\nUsing first available input device: {dev['name']}")
                            break

            if working_device is None or device_info is None:
                raise RuntimeError("No suitable input device found.")

            rate = int(device_info['default_samplerate'])
            needs_resampling = (rate != self.DESIRED_RATE)

            print("\nSelected device details:")
            print(f"  Name: {device_info['name']}")
            print(f"  Input channels: {device_info['max_input_channels']}")
            print(f"  Default samplerate: {rate}")
            print(f"  Low latency: {device_info['default_low_input_latency']}")
            print(f"  High latency: {device_info['default_high_input_latency']}")

            # Ensure sd.rec() uses the selected device
            sd.default.device = (working_device, None)

            # Grab a 4-second buffer for calibration
            calibration_duration = 4.0
            frames_needed = int(rate * calibration_duration)

            print(f"\nCalibrating noise floor for {calibration_duration}s. Please remain silent...")
            audio_buffer = sd.rec(frames_needed, samplerate=rate,
                                  channels=1, dtype='float32')
            sd.wait()  # blocks until recording finishes

            audio_buffer = audio_buffer.flatten()

            # Build chunk-based RMS
            chunk_size = 1024
            chunk_rms_list = []
            for i in range(0, len(audio_buffer), chunk_size):
                block = audio_buffer[i : i + chunk_size]
                if len(block) > 0:
                    block_rms_db = self._rms_db(block)
                    chunk_rms_list.append(block_rms_db)

            if not chunk_rms_list:
                raise ValueError("No samples captured during calibration.")

            sorted_rms = np.sort(chunk_rms_list)
            cutoff_index = int(0.90 * len(sorted_rms))
            quiet_rms = sorted_rms[:cutoff_index]

            if len(quiet_rms) < 10:
                print(f"\nWARNING: Only found {len(quiet_rms)} 'quiet' frames.")
                print("Using fallback noise floor of -55.0 dB.")
                initial_floor = -55.0
            else:
                initial_floor = float(np.percentile(quiet_rms, 20))
                print(f"\nCollected {len(chunk_rms_list)} total frames; {len(quiet_rms)} deemed quiet.")
                print(f"Robust noise floor estimate (20th percentile): {initial_floor:.1f} dB")

            # Optionally clamp extreme out-of-range floors
            if initial_floor < -120.0 or initial_floor > 0.0:
                print("Calibrated floor out of typical range, using fallback -55.0 dB.")
                initial_floor = -55.0

            self.noise_floor = initial_floor
            self.min_floor   = initial_floor
            self.max_floor   = initial_floor + 60
            self.rms_level   = initial_floor
            self.peak_level  = initial_floor

            print(f"\nMicrophone calibration successful:")
            print(f"  Noise Floor: {self.noise_floor:.1f} dB")
            print(f"  Floor range: {self.min_floor:.1f} to {self.max_floor:.1f} dB")
            print(f"  Sample rate: {rate} Hz")

            # Now open the main stream for continuous capture
            print("\nOpening main stream for recording...")
            stream = sd.InputStream(
                device=working_device,
                channels=1,
                samplerate=rate,
                dtype=np.float32,
                blocksize=self.CHUNK,
                latency='low'
            )
            stream.start()

            self.stream = stream
            self.rate = rate
            self.needs_resampling = needs_resampling

            return stream, device_info, rate, needs_resampling

        except Exception as e:
            raise RuntimeError(f"Failed to initialize audio: {str(e)}")

    def _rms_db(self, float_block):
        """Compute RMS in dB for a float32 block."""
        if len(float_block) == 0:
            return -96.0
        rms = np.sqrt(np.mean(float_block ** 2))
        return 20.0 * np.log10(max(rms, 1e-10))

    def bytes_to_float32_audio(self, audio_data, sample_rate=None):
        """
        Convert int16-encoded bytes to float32 samples in [-1..1].
        Returns (audio_float32, sample_rate).
        """
        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        return audio_float32, (sample_rate if sample_rate is not None else self.DESIRED_RATE)

    def process_audio(self, audio_data):
        """
        Uses end_silence_duration from config to determine how much silence
        must be detected to confirm speech has ended.
        """
        if len(audio_data) == 0:
            return {
                'audio': np.array([]),
                'is_speech': False,
                'db_level': self.rms_level,
                'noise_floor': self.noise_floor,
                'speech_ratio': 0.0,
                'zero_crossings': 0.0,
                'peak_level': self.peak_level
            }

        # 1) DC removal & pre-emphasis
        dc_removed = audio_data - np.mean(audio_data)
        emphasized = np.zeros_like(dc_removed)
        if len(dc_removed) > 0:
            emphasized[0] = dc_removed[0] - self.pre_emphasis * self.prev_sample
            if len(dc_removed) > 1:
                emphasized[1:] = dc_removed[1:] - self.pre_emphasis * dc_removed[:-1]
            self.prev_sample = dc_removed[-1]

        # 2) Envelope (RMS & peak)
        now = time.time()
        dt = now - self.last_update
        self.last_update = now

        block_rms = np.sqrt(np.mean(emphasized**2))
        block_peak = np.max(np.abs(emphasized))
        instant_rms_db = 20 * np.log10(max(block_rms, 1e-10))
        instant_peak_db = 20 * np.log10(max(block_peak, 1e-10))

        # Attack/Release for RMS
        if instant_rms_db > self.rms_level:
            alpha = 1.0 - np.exp(-dt / self.rms_attack)
        else:
            alpha = 1.0 - np.exp(-dt / self.rms_release)
        self.rms_level += (instant_rms_db - self.rms_level) * alpha

        # Attack/Release for Peak
        if instant_peak_db > self.peak_level:
            alpha = 1.0 - np.exp(-dt / self.peak_attack)
        else:
            alpha = 1.0 - np.exp(-dt / self.peak_release)
        self.peak_level += (instant_peak_db - self.peak_level) * alpha

        self.level_history.append(self.rms_level)

        # 3) Spectral ratio & gating thresholds
        speech_ratio, zero_crossings = self._spectral_analysis(emphasized)
        level_above_floor = self.rms_level - self.noise_floor

        open_threshold  = self.speech_threshold_open   # from config
        close_threshold = self.speech_threshold_close  # from config

        # Initialize state tracking if needed
        if not hasattr(self, 'silence_time'):
            self.silence_time = 0.0

        # Get required silence duration from config
        end_silence = self.config["speech_detection"]["end_silence_duration"]

        # Detect definite speech with more lenient conditions
        is_definite_speech = (
            level_above_floor > open_threshold or  # High level
            speech_ratio > self.ratio_override     # Very strong speech characteristics
        )

        # Detect probable speech with looser conditions
        is_probable_speech = (
            level_above_floor > (close_threshold + 2.0) and  # Above close threshold with margin
            speech_ratio > (self.close_ratio * 1.2)          # Above close ratio with margin
        )

        # Handle state transitions
        if not self.is_speaking:
            if is_definite_speech:
                self.is_speaking = True
                self.silence_time = 0.0
                print(f"Speech START - RMS: {self.rms_level:.1f} dB, Floor: {self.noise_floor:.1f} dB, Ratio: {speech_ratio:.3f}")
        else:
            # Already speaking - check for silence or continued speech
            if is_definite_speech or is_probable_speech:
                # Any kind of speech resets silence counter
                self.silence_time = 0.0
            else:
                # Accumulate silence time if no speech detected
                self.silence_time += dt
                # End speech only after enough silence
                if self.silence_time >= end_silence:
                    self.is_speaking = False
                    self.silence_time = 0.0
                    print(f"Speech END - RMS: {self.rms_level:.1f} dB, Floor: {self.noise_floor:.1f} dB, Ratio: {speech_ratio:.3f}")

        # 5) Debug every 50 frames
        if self.debug_counter % 50 == 0:
            print(f"\n[DEBUG] RMS: {self.rms_level:.1f} dB,"
                  f" Floor: {self.noise_floor:.1f} dB,"
                  f" AboveFloor: {level_above_floor:.1f} dB,"
                  f" Ratio: {speech_ratio:.3f},"
                  f" SilenceTime: {self.silence_time:.3f}s")
        self.debug_counter += 1

        return {
            'audio': emphasized,
            'is_speech': self.is_speaking,
            'db_level': self.rms_level,
            'noise_floor': self.noise_floor,
            'speech_ratio': speech_ratio,
            'zero_crossings': zero_crossings,
            'peak_level': self.peak_level
        }

    def _spectral_analysis(self, float_block):
        """
        Compute a speech ratio (speech-band energy vs. total) plus zero-crossing rate.
        """
        if len(float_block) == 0:
            return 0.0, 0.0
        try:
            nperseg = min(256, len(float_block))
            noverlap = nperseg // 2
            freqs, _, Sxx = signal.spectrogram(
                float_block,
                fs=16000,
                nperseg=nperseg,
                noverlap=noverlap,
                scaling='spectrum'
            )
            speech_mask = (freqs >= 100) & (freqs <= 3500)
            if Sxx.size > 0:
                speech_energy = np.mean(Sxx[speech_mask, :], axis=1)
                total_energy  = np.mean(Sxx, axis=0)
                ratio = (
                    np.mean(speech_energy) / np.mean(total_energy)
                    if np.mean(total_energy) > 0 else 0.0
                )
            else:
                ratio = 0.0

            # Zero-crossing rate
            zc = np.sum(np.abs(np.diff(np.signbit(float_block)))) / len(float_block)
            return ratio, zc

        except Exception:
            return 0.0, 0.0

    def close(self):
        """Close the audio stream if open."""
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            except Exception as e:
                print(f"Error closing audio stream: {e}")

    def calculate_volume(self, audio_data):
        """
        Example method to calculate volume in a compressed scale [0..1].
        Not used by gating, but can be used to display a meter in a GUI.
        """
        if self.rms_level is None or self.noise_floor is None or self.max_floor is None:
            return 0.0

        current_floor = self.noise_floor_ema
        if self.rms_level > current_floor:
            db_above_floor = float(self.rms_level - self.noise_floor)
            ratio = 0.8
            knee = 6.0
            if db_above_floor < -knee / 2:
                gain = db_above_floor
            elif db_above_floor > knee / 2:
                gain = -knee / 2 + (db_above_floor - (-knee / 2)) / ratio
            else:
                gain = db_above_floor + ((1 / ratio - 1) *
                        (db_above_floor + knee / 2)**2 / (2 * knee))
            volume = np.power(10, gain / 20) / np.power(10, (self.max_floor - self.noise_floor) / 20)
            return max(0.05, min(1.0, volume))
        return 0.0
