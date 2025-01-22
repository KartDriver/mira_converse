import sounddevice as sd
import numpy as np
import threading
from collections import deque
import time
from scipy import signal

class AudioOutput:
    def __init__(self):
        self.stream = None
        self.device_rate = None
        self.input_rate = 24000  # TTS output rate
        self.audio_queue = deque()  # Remove maxlen to preserve all chunks
        self.playing = False
        self.play_thread = None

    def _find_output_device(self):
        """Find a suitable audio output device"""
        try:
            # Print available devices for debugging
            print("\nAvailable output devices:")
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev['max_output_channels'] > 0:
                    print(f"Device {i}: {dev['name']} (outputs: {dev['max_output_channels']})")

            # Get default output device
            device_info = sd.query_devices(kind='output')
            print(f"\nSelected output device: {device_info['name']}")
            return device_info

        except Exception as e:
            print(f"Error finding output device: {e}")
            raise

    async def initialize(self):
        """Initialize audio output"""
        if self.stream and self.stream.active:
            return

        try:
            print("[TTS Output] Initializing audio output...")
            
            # Find suitable output device
            device_info = self._find_output_device()
            self.device_rate = int(device_info['default_samplerate'])
            
            # Create output stream with moderate latency
            self.stream = sd.OutputStream(
                samplerate=self.device_rate,
                channels=1,
                dtype=np.float32,
                latency='low'  # Keep low latency for responsiveness
            )
            self.stream.start()
            
            print("[TTS Output] Successfully initialized audio")
            
        except Exception as e:
            print(f"[TTS Output] Error initializing audio: {e}")
            if self.stream:
                self.stream.close()
                self.stream = None

    def _play_audio_thread(self):
        """Background thread for continuous audio playback"""
        while self.playing:
            try:
                if self.audio_queue:
                    chunk = self.audio_queue.popleft()
                    
                    # Remove TTS: prefix if present
                    if chunk.startswith(b'TTS:'):
                        chunk = chunk[4:]
                    
                    # Convert to float32 with clipping for cleaner audio
                    audio_data = np.clip(
                        np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0,
                        -1.0, 1.0
                    )
                    
                    # Resample if needed using resample_poly for better quality
                    if self.device_rate != self.input_rate:
                        # Calculate resampling parameters
                        gcd_val = np.gcd(self.device_rate, self.input_rate)
                        up = self.device_rate // gcd_val
                        down = self.input_rate // gcd_val
                        # Use resample_poly with small chunk size for better real-time performance
                        audio_data = signal.resample_poly(audio_data, up, down, padtype='line')
                    
                    # Play audio
                    if self.stream and self.stream.active:
                        self.stream.write(audio_data)
                else:
                    # Brief sleep when queue is empty
                    time.sleep(0.001)  # Keep original sleep time for responsiveness
                    
            except Exception as e:
                print(f"[TTS Output] Error in playback thread: {e}")
                time.sleep(0.001)  # Keep original error sleep time

    async def play_chunk(self, chunk):
        """Queue an audio chunk for playback"""
        try:
            self.audio_queue.append(chunk)
        except Exception as e:
            print(f"[TTS Output] Error queueing chunk: {e}")

    async def start_stream(self):
        """Start the audio stream and playback thread"""
        try:
            if not self.stream or not self.stream.active:
                await self.initialize()
            
            # Start playback thread if not already running
            if not self.playing:
                self.playing = True
                self.play_thread = threading.Thread(target=self._play_audio_thread)
                self.play_thread.daemon = True
                self.play_thread.start()
                print("[TTS Output] Playback thread started")
                
        except Exception as e:
            print(f"[TTS Output] Error starting stream: {e}")

    def pause(self):
        """Stop audio playback"""
        self.playing = False
        if self.play_thread:
            self.play_thread.join(timeout=1.0)
        if self.stream and self.stream.active:
            self.stream.stop()
        # Clear any remaining audio
        self.audio_queue.clear()

    def close(self):
        """Clean up audio resources"""
        self.playing = False
        if self.play_thread:
            self.play_thread.join(timeout=1.0)
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
