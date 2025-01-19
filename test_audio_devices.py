import pyaudio
import numpy as np
import time

def generate_sine_wave(frequency=440, duration=1, sample_rate=48000):
    """Generate a sine wave at the given frequency"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t)
    return (tone * 32767).astype(np.int16)

def test_audio_device(p, device_info, test_tone, rate=48000, channels=2):
    """Test audio playback on a specific device"""
    try:
        print(f"\nTesting device: {device_info['name']}")
        print(f"Index: {device_info['index']}")
        print(f"Max output channels: {device_info['maxOutputChannels']}")
        print(f"Testing with: {channels} channels at {rate}Hz")
        
        if device_info['maxOutputChannels'] == 0:
            print("This device has no output channels, skipping...")
            return
            
        # Convert mono test tone to stereo if needed
        if channels == 2:
            stereo_tone = np.column_stack((test_tone, test_tone))
            test_data = stereo_tone
        else:
            test_data = test_tone
            
        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            output=True,
            output_device_index=device_info['index'],
            frames_per_buffer=1024
        )
        
        print("Playing test tone... (1 second)")
        stream.write(test_data.tobytes())
        stream.stop_stream()
        stream.close()
        print("Finished testing device")
        time.sleep(1.0)  # Longer pause between tests
        
    except Exception as e:
        print(f"Error testing device: {e}")

def test_specific_device(p, card_index, device_index):
    """Test a specific sound card with multiple configurations"""
    try:
        device_name = f"hw:{card_index},{device_index}"
        print(f"\n=== Testing {device_name} ===")
        
        # Get device info
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if f"hw:{card_index}" in str(info):
                device_info = info
                break
        else:
            print(f"Could not find device {device_name}")
            return
            
        # Test different configurations
        rates = [48000, 44100, 16000]
        channels_options = [2, 1]
        
        for rate in rates:
            test_tone = generate_sine_wave(440, 1, rate)
            for channels in channels_options:
                try:
                    test_audio_device(p, device_info, test_tone, rate, channels)
                except Exception as e:
                    print(f"Failed with {channels} channels at {rate}Hz: {e}")

    except Exception as e:
        print(f"Error testing device {device_name}: {e}")

def main():
    p = pyaudio.PyAudio()
    
    # Print all audio devices
    print("\nAvailable audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"Device {info['index']}: {info['name']} (outputs: {info['maxOutputChannels']})")
    
    print("\nTesting specific devices...")
    
    # Test HD-Audio Generic card 0
    print("\n=== Testing HD-Audio Generic (card 0) ===")
    test_specific_device(p, 0, 0)
    
    # Test HD-Audio Generic card 1 (ALC257)
    print("\n=== Testing HD-Audio Generic (card 1) ===")
    test_specific_device(p, 1, 0)
    
    # Test AMD audio device
    print("\n=== Testing AMD audio device (card 2) ===")
    test_specific_device(p, 2, 0)
    
    p.terminate()
    print("\nDevice testing complete!")

if __name__ == "__main__":
    main()
