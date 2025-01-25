"""
Audio core that uses Py-WebRTCVad for speech detection
instead of manual floor/ratio gating.

We keep:
- init_audio_device() for device selection & optional 'calibration'
- process_audio() for chunk-based reading

BUT:
- The actual "is_speech" gating is done by WebRTC VAD in 20ms frames.
- We keep a short "hangover" (end_silence_frames) so we don't cut speech abruptly.
"""

import json
import numpy as np
import time
from collections import deque
import sounddevice as sd
import webrtcvad  # <-- New dependency
import platform
import warnings

class AudioCore:
    def __init__(self):
        with open('config.json', 'r') as f:
            self.config = json.load(f)

        # Basic audio properties
        self.stream = None
        self.rate = None
        self.needs_resampling = None
        self.CHUNK = self.config['audio_processing']['chunk_size']
        self.CHANNELS = 1
        self.DESIRED_RATE = self.config['audio_processing']['desired_rate']

        # We'll still track some "floor" for your GUI, but it's not used by the VAD gating
        self._noise_floor = -96.0
        self._min_floor   = -96.0
        self._max_floor   = -36.0
        self._rms_level   = -96.0
        self._peak_level  = -96.0
        self.last_update  = time.time()
        self.debug_counter = 0

        # This "calibration" is mostly for display logs
        self.calibrated_floor = None

        # --------------- NEW: WebRTC VAD Setup ---------------
        # Create a WebRTC VAD instance. Mode can be 0-3 (3 = most aggressive).
        self.vad = webrtcvad.Vad(mode=2)

        # We'll feed the VAD 20ms frames at 16kHz, 16-bit mono
        self.VAD_FRAME_MS = 20
        self.VAD_FRAME_SIZE = int(0.02 * self.DESIRED_RATE)  # 20ms * 16k
        self._vad_buffer = bytearray()  # Holds leftover PCM between calls

        # State machine for hangover:
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0

        # Adjust these thresholds as needed:
        self.start_speech_frames = 2  # must see 'speech' in 2+ consecutive frames
        self.end_silence_frames  = int(
            self.config['speech_detection']['end_silence_duration'] / (self.VAD_FRAME_MS / 1000.0)
        )
        # e.g. if end_silence_duration=0.8, 0.8 / 0.02 = 40 frames

    # ----------------------------------------------------------------
    # Optional: Keep your old floor logic for logging or GUI meter
    # ----------------------------------------------------------------
    @property
    def noise_floor(self):
        return self._noise_floor

    @noise_floor.setter
    def noise_floor(self, value):
        try:
            if value is not None:
                self._noise_floor = float(value)
            else:
                self._noise_floor = -96.0
        except:
            self._noise_floor = -96.0

    @property
    def min_floor(self):
        return self._min_floor

    @min_floor.setter
    def min_floor(self, value):
        self._min_floor = float(value)

    @property
    def max_floor(self):
        return self._max_floor

    @max_floor.setter
    def max_floor(self, value):
        self._max_floor = float(value)

    @property
    def rms_level(self):
        return self._rms_level

    @rms_level.setter
    def rms_level(self, value):
        self._rms_level = float(value)

    @property
    def peak_level(self):
        return self._peak_level

    @peak_level.setter
    def peak_level(self, value):
        self._peak_level = float(value)

    # ----------------------------------------------------------------
    # Audio Device Initialization & Optional "Calibration"
    # ----------------------------------------------------------------
    def init_audio_device(self):
        """
        Initialize audio device. We might do a short "calibration" for your GUI's floor display,
        but we won't rely on it for gating. The gating is handled by WebRTC VAD.
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

            # Simple selection logic
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    # Example: prefer a name with 'microphone' on macOS
                    if system == 'darwin' and 'microphone' in device['name'].lower():
                        working_device = i
                        device_info = device
                        break
                    elif system == 'linux' and 'acp' in device['name'].lower():
                        working_device = i
                        device_info = device
                        break

            # fallback to default if none matched
            if working_device is None:
                default_idx = sd.query_default_device('input')[0]
                device_info = sd.query_devices(default_idx)
                working_device = default_idx
                print(f"\nUsing default input device: {device_info['name']}")

            if working_device is None or device_info is None:
                raise RuntimeError("No suitable input device found.")

            rate = int(device_info['default_samplerate'])
            needs_resampling = (rate != self.DESIRED_RATE)

            print("\nSelected device details:")
            print(f"  Name: {device_info['name']}")
            print(f"  Input channels: {device_info['max_input_channels']}")
            print(f"  Default samplerate: {rate}")
            print(f"  Latency: {device_info['default_low_input_latency']}")

            sd.default.device = (working_device, None)

            # If you still want a "floor" for the GUI, do a short capture (2s)
            # We won't rely on it for gating.
            calibration_duration = 2.0
            frames_needed = int(rate * calibration_duration)
            print(f"\nCalibrating GUI floor for {calibration_duration}s... (not used by VAD)")
            audio_buffer = sd.rec(frames_needed, samplerate=rate,
                                  channels=1, dtype='float32')
            sd.wait()
            audio_buffer = audio_buffer.flatten()

            chunk_rms_list = []
            chunk_size = 1024
            for i in range(0, len(audio_buffer), chunk_size):
                block = audio_buffer[i : i + chunk_size]
                if len(block) > 0:
                    block_rms = np.sqrt(np.mean(block**2))
                    block_rms_db = 20.0 * np.log10(max(block_rms, 1e-10))
                    chunk_rms_list.append(block_rms_db)

            if chunk_rms_list:
                initial_floor = float(np.percentile(chunk_rms_list, 20))
                # clamp
                if initial_floor < -85.0:
                    initial_floor = -85.0
                if initial_floor > -20.0:
                    initial_floor = -20.0
                self.noise_floor = initial_floor
                self.min_floor   = initial_floor
                self.max_floor   = initial_floor + 60
                self.rms_level   = initial_floor
                self.peak_level  = initial_floor
                self.calibrated_floor = initial_floor
            else:
                self.calibrated_floor = -60.0

            print(f"  GUI floor set to: {self.calibrated_floor:.1f} dB (not used by VAD)")

            # Open the main stream for continuous capture
            print("\nOpening main input stream...")
            stream = sd.InputStream(
                device=working_device,
                channels=1,
                samplerate=rate,
                dtype=np.float32,
                blocksize=self.CHUNK
            )
            stream.start()

            self.stream = stream
            self.rate = rate
            self.needs_resampling = needs_resampling

            return stream, device_info, rate, needs_resampling

        except Exception as e:
            raise RuntimeError(f"Failed to initialize audio: {str(e)}")

    # ----------------------------------------------------------------
    # Convert from float -> PCM int16 for VAD, etc.
    # ----------------------------------------------------------------
    def bytes_to_float32_audio(self, audio_data, sample_rate=None):
        """
        Convert int16-encoded bytes to float32 samples in [-1..1].
        This is typically what your client is sending/receiving.
        """
        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        return audio_float32, (sample_rate if sample_rate is not None else self.DESIRED_RATE)

    # ----------------------------------------------------------------
    # The "process_audio" function:
    # 1) Takes a float32 buffer
    # 2) Convert to 16-bit PCM
    # 3) Break into 20ms frames for WebRTC VAD
    # 4) Update speech state
    # 5) Return 'is_speech' (the overall gate state)
    # ----------------------------------------------------------------
    def process_audio(self, audio_data):
        """
        - We do RMS calculations for your GUI (optional)
        - Real gating uses WebRTC VAD in 20ms frames
        - We maintain a short state machine: is_speaking or not
        """
        # 1) For the GUI meter, let's compute RMS dB for the entire chunk
        if len(audio_data) > 0:
            block_rms = np.sqrt(np.mean(audio_data**2))
            instant_rms_db = 20.0 * np.log10(max(block_rms, 1e-10))

            # Attack/Release for your GUI's RMS
            now = time.time()
            dt = now - self.last_update
            self.last_update = now

            # Very simplistic:
            alpha = 0.5
            if instant_rms_db > self.rms_level:
                self.rms_level += alpha * (instant_rms_db - self.rms_level)
            else:
                self.rms_level += alpha * (instant_rms_db - self.rms_level)

            if instant_rms_db > self.peak_level:
                self.peak_level = instant_rms_db
            else:
                self.peak_level *= 0.99  # slow decay
        else:
            # no data
            pass

        # 2) Convert float -> int16 -> bytes so WebRTC VAD can handle it
        int16_data = np.clip(audio_data * 32767.0, -32767, 32767).astype(np.int16)
        pcm_bytes = int16_data.tobytes()

        # 3) Feed it into our buffer, chunk out 20ms frames
        self._vad_buffer.extend(pcm_bytes)

        # We'll define an "is_speech" for the final chunk. We might set it to self.is_speaking
        # after we process everything.
        # But note: one chunk from your mic might be 128ms, we break it into multiple 20ms frames.
        while len(self._vad_buffer) >= (self.VAD_FRAME_SIZE * 2):  # 2 bytes per sample
            frame = self._vad_buffer[:(self.VAD_FRAME_SIZE * 2)]
            self._vad_buffer = self._vad_buffer[(self.VAD_FRAME_SIZE * 2):]

            # 4) Check VAD
            #    The sample rate for VAD must be 16000 if your desired rate is 16k
            #    If you truly need 16k, ensure your device is giving that or resample if needed
            try:
                webrtc_is_speech = self.vad.is_speech(frame, sample_rate=16000)
            except Exception as e:
                # If there's any error, treat it as silence
                webrtc_is_speech = False

            # 5) Update state machine
            if webrtc_is_speech:
                self.speech_frames += 1
                self.silence_frames = 0
            else:
                self.silence_frames += 1
                self.speech_frames = 0

            if not self.is_speaking:
                # If we accumulate enough consecutive "speech" frames, start
                if self.speech_frames >= self.start_speech_frames:
                    self.is_speaking = True
                    # Optionally log
                    print(f"Speech START - frames:{self.speech_frames}")
            else:
                # We are speaking, check if we get enough silence frames
                if self.silence_frames >= self.end_silence_frames:
                    self.is_speaking = False
                    # Optionally log
                    print(f"Speech END - silence_frames:{self.silence_frames}")

        # We'll let the rest remain in the buffer until we get enough for the next frame
        # Return is_speech for the last chunk
        # But typically you'd keep track of self.is_speaking at the server or client
        return {
            'audio': audio_data,    # The processed float data if you need it
            'is_speech': self.is_speaking,
            'db_level': self.rms_level,
            'noise_floor': self.noise_floor,  # for GUI
            'peak_level': self.peak_level
        }

    # ----------------------------------------------------------------
    # If the server or client needs to close resources
    # ----------------------------------------------------------------
    def close(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            except Exception as e:
                print(f"Error closing audio stream: {e}")
