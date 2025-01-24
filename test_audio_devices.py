#!/usr/bin/env python3
"""
Simple test script to play a beep sound through the default audio device.
"""

import sounddevice as sd
import numpy as np
import time

def generate_stereo_beep(duration=1.0, left_freq=440.0, right_freq=880.0, amplitude=0.8):
    """Generate a stereo beep with different frequencies for left and right channels"""
    sample_rate = 24000  # Match TTS output rate
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate different tones for left and right
    left_channel = amplitude * np.sin(2 * np.pi * left_freq * t)
    right_channel = amplitude * np.sin(2 * np.pi * right_freq * t)
    
    # Apply fade in/out to avoid clicks
    fade_len = int(0.05 * sample_rate)  # 50ms fade
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    
    left_channel[:fade_len] *= fade_in
    left_channel[-fade_len:] *= fade_out
    right_channel[:fade_len] *= fade_in
    right_channel[-fade_len:] *= fade_out
    
    # Combine into stereo array
    stereo = np.column_stack((left_channel, right_channel))
    return stereo.astype(np.float32)

def test_device(device_idx, beep_data, input_rate=24000):
    """Test a specific audio device"""
    try:
        # Get device info
        device_info = sd.query_devices(device_idx)
        print(f"\nTesting device: {device_info['name']}")
        print(f"Max output channels: {device_info['max_output_channels']}")
        print(f"Device sample rate: {device_info['default_samplerate']}Hz")
        
        # Get device's native sample rate
        device_rate = int(device_info['default_samplerate'])
        
        # Resample if needed
        if device_rate != input_rate:
            print(f"Resampling from {input_rate}Hz to {device_rate}Hz")
            from scipy import signal
            
            # Resample each channel separately
            ratio = device_rate / input_rate
            num_samples = int(beep_data.shape[0] * ratio)
            
            resampled_left = signal.resample(beep_data[:, 0], num_samples)
            resampled_right = signal.resample(beep_data[:, 1], num_samples)
            
            beep_data = np.column_stack((resampled_left, resampled_right))
        
        # Create output stream
        print("Opening audio stream...")
        with sd.OutputStream(
            device=device_idx,
            samplerate=device_rate,  # Use device's native rate
            channels=2,              # Stereo output
            dtype=np.float32,
            latency='low'
        ) as stream:
            print("Stream opened successfully")
            print("Playing beep...")
            stream.write(beep_data)
            time.sleep(1.5)
            print("Beep played successfully")
            return True
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Main test function"""
    try:
        # Print available devices for debugging
        print("\nAvailable output devices:")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_output_channels'] > 0:
                print(f"Device {i}: {dev['name']} (outputs: {dev['max_output_channels']}, rate: {dev['default_samplerate']}Hz)")
        
        # Generate test sound
        print("\nGenerating stereo test beep...")
        beep = generate_stereo_beep(duration=2.0)  # Longer duration to better hear the effect
        
        # Try PipeWire device first (usually most compatible)
        pipewire_idx = None
        for i, dev in enumerate(devices):
            if 'pipewire' in dev['name'].lower():
                pipewire_idx = i
                break
        
        if pipewire_idx is not None:
            print("\nTrying PipeWire device...")
            if test_device(pipewire_idx, beep):
                return
        
        # Fall back to default device if PipeWire failed
        print("\nTrying default device...")
        default_idx = sd.default.device[1]
        test_device(default_idx, beep)

    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        print("\nTest complete")
