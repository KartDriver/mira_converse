import numpy as np
import subprocess
import tempfile
import os
import soundfile as sf

class AudioOutput:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.current_process = None
        
    def initialize(self):
        pass
            
    def start_stream(self):
        pass

    def play_chunk(self, chunk):
        """Play an audio chunk using pw-play"""
        try:
            # Convert bytes to audio data (24kHz int16 PCM)
            audio_data = np.frombuffer(chunk, dtype=np.int16)
            
            # Save to temporary WAV file
            temp_path = os.path.join(self.temp_dir, 'temp_audio.wav')
            sf.write(temp_path, audio_data, 24000)  # TTS output is at 24kHz
            
            # Play using pw-play
            subprocess.run(['pw-play', '--volume=1.0', temp_path], check=True)
            
        except Exception as e:
            print(f"[TTS Playback Error] {str(e)}")
            
    def pause(self):
        pass

    def close(self):
        """Clean up temporary directory"""
        try:
            temp_path = os.path.join(self.temp_dir, 'temp_audio.wav')
            if os.path.exists(temp_path):
                os.remove(temp_path)
            os.rmdir(self.temp_dir)
        except Exception as e:
            print(f"[TTS Playback] Error cleaning up: {e}")
