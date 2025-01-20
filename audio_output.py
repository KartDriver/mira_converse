import numpy as np
import pyaudio
import threading
from collections import deque
import samplerate
import time
import platform

class AudioOutput:
    def __init__(self):
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = deque()
        self.playing = False
        self.play_thread = None
        
    def _find_output_device(self):
        """Find a suitable audio output device"""
        system = platform.system().lower()
        
        # Print available devices for debugging
        print("\nAvailable output devices:")
        for i in range(self.pyaudio.get_device_count()):
            try:
                dev_info = self.pyaudio.get_device_info_by_index(i)
                print(f"Device {i}: {dev_info['name']} (outputs: {dev_info['maxOutputChannels']})")
            except Exception as e:
                print(f"Error getting device {i} info: {e}")
        
        # Try system default output device first
        try:
            default_device = self.pyaudio.get_default_output_device_info()
            if default_device['maxOutputChannels'] > 0:
                print(f"\nSelected default output device: {default_device['name']}")
                return default_device
        except Exception:
            pass

        # Fall back to first device with output channels
        for i in range(self.pyaudio.get_device_count()):
            try:
                device_info = self.pyaudio.get_device_info_by_index(i)
                if device_info['maxOutputChannels'] > 0:
                    print(f"\nSelected output device: {device_info['name']}")
                    return device_info
            except Exception:
                continue
                
        raise RuntimeError("No suitable output device found")
        
    def initialize(self):
        """Initialize audio output stream"""
        if self.stream:  # Already initialized
            return
            
        try:
            print("[TTS Output] Initializing audio output...")
            
            # Find suitable output device
            device_info = self._find_output_device()
            print(f"[TTS Output] Using device: {device_info['name']}")
            
            # Configure stream based on OS
            system = platform.system().lower()
            stream_params = {
                'format': pyaudio.paInt16,  # Use int16 for all platforms
                'channels': 1,  # Mono
                'rate': 24000,  # Match TTS output rate
                'output': True,
                'start': False,
                'output_device_index': device_info['index']
            }
            
            # OS-specific configurations
            if system == 'darwin':
                # macOS: smaller buffer for lower latency
                stream_params['frames_per_buffer'] = 1024
            elif system == 'linux':
                # Linux: larger buffer for stability
                stream_params['frames_per_buffer'] = 2048
                # Let ALSA handle buffer management
                stream_params['input_host_api_specific_stream_info'] = None
            
            # Open stream with configured parameters
            self.stream = self.pyaudio.open(**stream_params)
            
            print("[TTS Output] Successfully initialized audio")
            
        except Exception as e:
            print(f"[TTS Output] Error initializing audio: {e}")
            
    def start_stream(self):
        """Start the audio stream and playback thread"""
        try:
            if not self.stream:
                self.initialize()
            
            # Ensure stream is started
            if not self.stream.is_active():
                self.stream.start_stream()
                print("[TTS Output] Audio stream started")
            
            # Start playback thread if not already running
            if not self.playing:
                self.playing = True
                self.play_thread = threading.Thread(target=self._play_audio_thread)
                self.play_thread.daemon = True
                self.play_thread.start()
                print("[TTS Output] Playback thread started")
                
        except Exception as e:
            print(f"[TTS Output] Error starting stream: {e}")
            
    def _play_audio_thread(self):
        """Background thread for continuous audio playback"""
        while self.playing:
            try:
                # Use a shorter timeout to be more responsive
                if self.audio_queue:
                    chunk = self.audio_queue.popleft()
                    
                    # Remove TTS: prefix if present
                    if chunk.startswith(b'TTS:'):
                        chunk = chunk[4:]
                    
                    # Convert audio data and play directly at 24kHz
                    audio_data = np.frombuffer(chunk, dtype=np.int16)
                    
                    if not self.stream.is_active():
                        self.stream.start_stream()
                    
                    self.stream.write(audio_data.tobytes())
                else:
                    # Minimal sleep when queue is empty
                    time.sleep(0.0001)  # 100 microseconds
                    
            except Exception as e:
                print(f"[TTS Output] Error in playback thread: {e}")
                # Brief pause on error before retrying
                time.sleep(0.001)

    def play_chunk(self, chunk):
        """Queue an audio chunk for playback"""
        try:
            # Add chunk to queue immediately without checking stream
            # (stream will already be started from trigger detection)
            self.audio_queue.append(chunk)
        except Exception as e:
            print(f"[TTS Output] Error queueing chunk: {e}")
            
    def pause(self):
        """Stop audio playback"""
        self.playing = False
        if self.play_thread:
            self.play_thread.join(timeout=1.0)
        if self.stream:
            self.stream.stop_stream()
        # Clear any remaining audio in queue
        self.audio_queue.clear()
        
    def close(self):
        """Clean up audio resources"""
        self.playing = False
        if self.play_thread:
            self.play_thread.join(timeout=1.0)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio.terminate()
