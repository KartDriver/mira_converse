import pyaudio
import numpy as np
import time
import sys

def generate_sine_wave(frequency=440, duration=1, sample_rate=48000):
    """Generate a sine wave at the given frequency"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t)
    return (tone * 32767).astype(np.int16)

def test_laptop_speakers():
    p = pyaudio.PyAudio()
    
    try:
        # Find the ALC257 device
        alc257_index = None
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if 'ALC257' in info['name']:
                alc257_index = i
                break
        
        if alc257_index is None:
            print("Could not find ALC257 device")
            return
            
        print(f"\nTesting laptop speakers (ALC257)...")
        print(f"Device index: {alc257_index}")
        
        # Generate test tones at different frequencies
        frequencies = [440, 880, 1760]  # A4, A5, A6
        sample_rate = 48000
        
        # Open stream with stereo output
        stream = p.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=sample_rate,
            output=True,
            output_device_index=alc257_index,
            frames_per_buffer=1024
        )
        
        for freq in frequencies:
            print(f"\nPlaying {freq}Hz tone...")
            tone = generate_sine_wave(freq, 1, sample_rate)
            # Convert to stereo
            stereo_tone = np.column_stack((tone, tone))
            stream.write(stereo_tone.tobytes())
            time.sleep(0.5)  # Pause between tones
            
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        p.terminate()

if __name__ == "__main__":
    print("Make sure no other audio applications are running...")
    time.sleep(2)
    test_laptop_speakers()
    print("\nTest complete!")
