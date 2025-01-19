import pyaudio
import numpy as np
import time

def test_audio():
    p = pyaudio.PyAudio()
    
    # List all audio devices
    print("\nAvailable Audio Devices:")
    for i in range(p.get_device_count()):
        try:
            dev_info = p.get_device_info_by_index(i)
            print(f"Device {i}: {dev_info['name']}")
            print(f"  Max Output Channels: {dev_info['maxOutputChannels']}")
            print(f"  Default Sample Rate: {dev_info['defaultSampleRate']}")
        except Exception as e:
            print(f"Error getting device {i} info: {e}")
    
    try:
        # Try to find the acp63 device
        device_index = None
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            if 'acp63' in dev_info['name'].lower():
                print(f"\nFound AMD audio device: {dev_info['name']}")
                device_index = i
                break
                
        if device_index is None:
            raise Exception("Could not find acp63 device")
            
        device_info = p.get_device_info_by_index(device_index)
        print(f"\nUsing device: {device_info['name']}")
        print(f"Sample rate: 48000 Hz (forced)")
        
        # Generate a simple beep
        sample_rate = 48000  # Force 48kHz for AMD audio
        duration = 1  # seconds
        frequency = 440  # Hz
        samples = np.arange(duration * sample_rate)
        signal = np.sin(2 * np.pi * frequency * samples / sample_rate)
        audio_data = (signal * 32767).astype(np.int16)
        
        # Open stream with minimal configuration
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            output=True,
            output_device_index=device_index,
            frames_per_buffer=1024,
            start=False  # Don't start yet
        )
        
        # Start stream explicitly
        stream.start_stream()
        
        print("\nTrying to play a test tone...")
        # Play the sound
        stream.write(audio_data.tobytes())
        time.sleep(1)
        
        # Cleanup
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"Error during playback: {e}")
    finally:
        p.terminate()

if __name__ == "__main__":
    test_audio()
