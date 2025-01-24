import sounddevice as sd
import numpy as np
import threading
from collections import deque
import time
from scipy import signal
import platform

class AudioOutput:
    def __init__(self):
        self.stream = None
        self.device_rate = None
        self.input_rate = 24000  # TTS output rate
        self.audio_queue = deque()  # Remove maxlen to preserve all chunks
        self.playing = False
        self.play_thread = None
        self.current_device = None

    def _find_output_device(self, device_name=None):
        """
        Find a suitable audio output device with improved Linux compatibility.
        
        Args:
            device_name: Optional name of device to use. If None, uses system default.
            
        Returns:
            Tuple of (device_index, device_info)
        """
        try:
            # Print available devices for debugging
            print("\nAvailable output devices:")
            devices = sd.query_devices()
            output_devices = []
            
            # First pass: collect all output devices
            for i, dev in enumerate(devices):
                if dev['max_output_channels'] > 0:
                    print(f"Device {i}: {dev['name']} (outputs: {dev['max_output_channels']}, rate: {dev['default_samplerate']}Hz)")
                    output_devices.append((i, dev))

            # Try to find PipeWire device first on Linux
            if platform.system() == 'Linux':
                # First try PipeWire
                for i, dev in output_devices:
                    if 'pipewire' in dev['name'].lower():
                        print(f"\nSelected PipeWire device: {dev['name']}")
                        self.current_device = dev
                        return i, dev
                
                # Then try other Linux-specific devices
                preferred_keywords = [
                    'pulse', 'default',  # Modern audio servers
                    'hw:', 'plughw:',    # ALSA devices
                    'dmix', 'surround'   # ALSA plugins
                ]
                
                # Try to find device by name if provided
                if device_name:
                    for i, dev in output_devices:
                        if dev['max_output_channels'] > 0 and dev['name'] == device_name:
                            print(f"\nSelected output device: {dev['name']}")
                            self.current_device = dev
                            return i, dev
                
                # Otherwise try preferred devices in order
                for keyword in preferred_keywords:
                    for i, dev in output_devices:
                        if keyword in dev['name'].lower():
                            print(f"\nSelected Linux output device: {dev['name']}")
                            self.current_device = dev
                            return i, dev
            
            # For non-Linux systems or if no preferred device found
            if device_name:
                for i, dev in output_devices:
                    if dev['max_output_channels'] > 0 and dev['name'] == device_name:
                        print(f"\nSelected output device: {dev['name']}")
                        self.current_device = dev
                        return i, dev

            # Fall back to default output device
            default_idx = sd.default.device[1]  # Get default output device index
            if default_idx is not None and 0 <= default_idx < len(devices):
                device_info = devices[default_idx]
            else:
                # If default device index is invalid, use first available output device
                device_info = next((dev for i, dev in output_devices), None)
                if device_info is None:
                    raise RuntimeError("No output devices found")
                default_idx = output_devices[0][0]
            
            print(f"\nSelected default output device: {device_info['name']}")
            self.current_device = device_info
            return default_idx, device_info

        except Exception as e:
            print(f"Error finding output device: {e}")
            raise

    def get_device_name(self):
        """Get the name of the current output device"""
        if self.current_device:
            return self.current_device['name']
        return "No device selected"

    def set_device_by_name(self, device_name):
        """
        Change the output device by name.
        
        Args:
            device_name: Name of the device to use
        """
        print(f"\nChanging output device to: {device_name}")
        try:
            # Stop current playback
            self.pause()
            if self.stream:
                self.stream.close()
                self.stream = None
            
            # Find and set new device
            device_idx, device_info = self._find_output_device(device_name)
            self.device_rate = int(device_info['default_samplerate'])
            
            # Create output stream
            self.stream = sd.OutputStream(
                device=device_idx,
                samplerate=self.device_rate,
                channels=1,
                dtype=np.float32,
                latency='low'
            )
            self.stream.start()
            print(f"Successfully switched to device: {device_name}")
            
        except Exception as e:
            print(f"Error changing output device: {e}")
            # Try to fall back to default device
            self.initialize_sync()

    def initialize_sync(self):
        """Initialize audio output synchronously"""
        if self.stream and self.stream.active:
            return

        try:
            print("[TTS Output] Initializing audio output...")
            
            # Find suitable output device
            device_idx, device_info = self._find_output_device()
            self.device_rate = int(device_info['default_samplerate'])
            
            # Create output stream
            self.stream = sd.OutputStream(
                device=device_idx,
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

    async def initialize(self):
        """Initialize audio output"""
        if self.stream and self.stream.active:
            return

        try:
            print("[TTS Output] Initializing audio output...")
            
            # Find suitable output device
            device_idx, device_info = self._find_output_device()
            self.device_rate = int(device_info['default_samplerate'])
            
            # Create output stream
            self.stream = sd.OutputStream(
                device=device_idx,
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
                self.playing = False # Ensure thread exits on error
                break # Exit thread on playback error

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
        """Clean up audio resources and ensure thread termination"""
        print("[TTS Output] AudioOutput.close() called - start") # Debug log
        self.playing = False
        # Signal the thread to stop
        if self.play_thread and self.play_thread.is_alive():
            try:
                self.play_thread.join(timeout=1.0)  # Wait for the thread to finish
            except Exception as e:
                print(f"[TTS Output] Error joining playback thread: {e}")

        # Close the stream
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"[TTS Output] Error closing stream: {e}")
            finally:
                self.stream = None
        print("[TTS Output] AudioOutput.close() finished") # Debug log
