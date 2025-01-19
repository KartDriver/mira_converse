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
            
            # Open stream with standard parameters
            self.stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,  # Mono
                rate=48000,  # Standard rate that most devices support
                output=True,
                start=False
            )
            
            print("[TTS Output] Successfully initialized audio")
            
        except Exception as e:
            print(f"[TTS Output] Error initializing audio: {e}")
            
    # def _test_audio(self):
    #     """Generate a short beep to test audio output"""
    #     try:
    #         # Generate a short 440Hz beep
    #         duration = 0.2  # seconds
    #         samples = np.arange(int(duration * 48000))
    #         test_audio = np.sin(2 * np.pi * 440 * samples / 48000)
    #         test_audio = (test_audio * 32767).astype(np.int16)
    #         
    #         print("[TTS Output] Testing audio with short beep...")
    #         self.stream.start_stream()
    #         self.stream.write(test_audio.tobytes())
    #         # Keep stream running for TTS
    #         
    #     except Exception as e:
    #         print(f"[TTS Output] Error testing audio: {e}")
    #     
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
                    
                    # Convert audio from 24kHz to 48kHz
                    audio_data = np.frombuffer(chunk, dtype=np.int16)
                    
                    # Simple linear interpolation
                    resampled = np.interp(
                        np.linspace(0, len(audio_data), len(audio_data) * 2),
                        np.arange(len(audio_data)),
                        audio_data
                    ).astype(np.int16)
                    
                    # Play the resampled audio immediately
                    if not self.stream.is_active():
                        self.stream.start_stream()
                    
                    self.stream.write(resampled.tobytes())
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
