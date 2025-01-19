import pyaudio
import platform
import warnings

# Use a standard ALSA-friendly buffer size (power of 2)
# 2048 samples gives us ~46ms at 44.1kHz which is a good balance
CHUNK = 2048
FORMAT = pyaudio.paInt16  # 16-bit PCM format
CHANNELS = 1
DESIRED_RATE = 16000      # Server expects 16kHz audio

def is_rate_supported(p, device_info, rate):
    """Check if the given sample rate is supported by the device"""
    try:
        return p.is_format_supported(
            rate,
            input_device=device_info['index'],
            input_channels=CHANNELS,
            input_format=FORMAT
        )
    except ValueError:
        return False

def find_input_device(p):
    """
    Find a suitable audio input device with the following priority:
    1. On macOS: Built-in microphone
    2. On Linux: Built-in microphone (typically ALC or similar)
    3. System default input device
    4. First available input device as fallback
    """
    system = platform.system().lower()
    
    # On Linux, try to suppress common ALSA warnings
    if system == 'linux':
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sounddevice")

    # Print detailed device information for debugging
    print("\nAvailable audio devices (detailed):")
    print("=" * 80)
    for i in range(p.get_device_count()):
        try:
            device_info = p.get_device_info_by_index(i)
            print(f"\nDevice {i}: {device_info['name']}")
            print(f"  Index: {device_info['index']}")
            print(f"  Input channels: {device_info['maxInputChannels']}")
            print(f"  Output channels: {device_info['maxOutputChannels']}")
            print(f"  Default sample rate: {device_info['defaultSampleRate']}")
            print(f"  Default input: {p.get_default_input_device_info()['index'] == device_info['index']}")
            print(f"  Host API: {p.get_host_api_info_by_index(device_info['hostApi'])['name']}")
            
            if device_info['maxInputChannels'] > 0:
                print("  Supported sample rates:")
                for rate in [16000, 22050, 32000, 44100, 48000]:
                    supported = is_rate_supported(p, device_info, rate)
                    print(f"    {rate} Hz: {'Yes' if supported else 'No'}")
            print("-" * 40)
        except OSError:
            print(f"Device {i}: <error accessing device>")

    # First try to find built-in microphone based on system
    for i in range(p.get_device_count()):
        try:
            device_info = p.get_device_info_by_index(i)
            device_name = device_info.get('name', '').lower()
            
            if device_info['maxInputChannels'] > 0:
                # On macOS, prefer the built-in microphone
                if system == 'darwin' and 'macbook' in device_name and 'microphone' in device_name:
                    print("\nSelected MacBook's built-in microphone")
                    return device_info
                    
                # On Linux, prioritize AMD audio device for Lenovo laptops
                elif system == 'linux':
                    # First preference: AMD audio device
                    if 'acp' in device_name:
                        print(f"\nSelected AMD audio device: {device_info['name']}")
                        return device_info
                    # Second preference: Default device
                    try:
                        default_info = p.get_default_input_device_info()
                        if default_info['index'] == device_info['index']:
                            print(f"\nSelected default ALSA device: {device_info['name']}")
                            return device_info
                    except OSError:
                        pass
        except OSError:
            continue

    # Try system default input device
    try:
        default_device_info = p.get_default_input_device_info()
        if default_device_info['maxInputChannels'] > 0:
            print("\nSelected system default input device")
            return default_device_info
    except OSError:
        pass

    # Fall back to first available input device
    for i in range(p.get_device_count()):
        try:
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                print("\nSelected first available input device")
                return device_info
        except OSError:
            continue
    
    raise RuntimeError("No suitable input device found. Please ensure you have a working microphone connected.")

def init_audio():
    """Initialize PyAudio with proper error handling"""
    p = pyaudio.PyAudio()
    try:
        # Print available devices for debugging
        print("\nAvailable audio devices:")
        for i in range(p.get_device_count()):
            try:
                dev_info = p.get_device_info_by_index(i)
                print(f"Device {i}: {dev_info['name']} (inputs: {dev_info['maxInputChannels']})")
            except OSError:
                print(f"Device {i}: <error accessing device>")

        # Find a suitable input device
        device_info = find_input_device(p)
        if device_info['maxInputChannels'] > 0:
            print(f"\nSelected audio device: {device_info['name']} (index: {device_info['index']})")
            
            # Check if device supports 16kHz directly
            rate = DESIRED_RATE
            needs_resampling = not is_rate_supported(p, device_info, DESIRED_RATE)
            
            if needs_resampling:
                # Find the closest supported rate above 16kHz
                supported_rates = [r for r in [16000, 22050, 32000, 44100, 48000] 
                                 if is_rate_supported(p, device_info, r)]
                if not supported_rates:
                    raise ValueError("No suitable sample rates supported by the device")
                rate = min(supported_rates)
                print(f"\nDevice doesn't support 16kHz directly. Using {rate}Hz with resampling.")
            else:
                print("\nDevice supports 16kHz recording directly.")
            
            return p, device_info, rate, needs_resampling
        else:
            raise ValueError("Selected device has no input channels")
    except Exception as e:
        p.terminate()
        raise RuntimeError(f"Failed to initialize audio: {str(e)}")
