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
        """Start the audio stream"""
        try:
            if not self.stream:
                self.initialize()
            if not self.stream.is_active():
                self.stream.start_stream()
        except Exception as e:
            print(f"[TTS Output] Error starting stream: {e}")
            
    def play_chunk(self, chunk):
        """Play an audio chunk directly"""
        try:
            if not self.stream:
                self.initialize()
                
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
            
            # Play the resampled audio
            if not self.stream.is_active():
                self.stream.start_stream()
            self.stream.write(resampled.tobytes())
            
        except Exception as e:
            print(f"[TTS Output] Error playing chunk: {e}")
            
    def pause(self):
        """Stop the audio stream"""
        if self.stream:
            self.stream.stop_stream()
        
    def close(self):
        """Clean up audio resources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio.terminate()
