import pyaudio
import numpy as np
import time

def print_device_info(p):
    """Print detailed info about all audio devices"""
    print("\nAudio Device Information:")
    print("-" * 50)
    
    try:
        default_device = p.get_default_output_device_info()
        print(f"\nDefault Output Device:")
        for key, value in default_device.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error getting default device: {e}")
    
    print("\nAll Available Devices:")
    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
            print(f"\nDevice {i}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error getting device {i} info: {e}")

def generate_test_tone(frequency=440, duration=3, sample_rate=48000, volume=0.5):
    """Generate a test tone with increasing volume"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Create a tone that increases in volume
    envelope = np.linspace(0, 1, len(t))
    tone = np.sin(2 * np.pi * frequency * t) * envelope * volume
    return (tone * 32767).astype(np.int16)

def main():
    p = pyaudio.PyAudio()
    
    try:
        # Print detailed device information
        print_device_info(p)
        
        # Generate test tone
        print("\nGenerating test tone...")
        sample_rate = 48000
        test_tone = generate_test_tone(sample_rate=sample_rate)
        
        # Convert to stereo
        stereo_tone = np.column_stack((test_tone, test_tone))
        
        print("\nOpening audio stream...")
        stream = p.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=sample_rate,
            output=True,
            frames_per_buffer=1024
        )
        
        print("\nPlaying test tone (3 seconds)...")
        stream.write(stereo_tone.tobytes())
        print("Finished playing test tone")
        
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        p.terminate()

if __name__ == "__main__":
    main()
