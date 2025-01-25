# Audio Chat

A real-time voice interaction system that combines speech recognition, natural language processing, and text-to-speech capabilities to create a seamless conversational AI experience.

## Features

- Real-time speech recognition using Whisper
- Natural language processing with LLM integration
- Text-to-speech synthesis using Kokoro
- Professional-grade audio processing with noise floor calibration
- Graphical interface for audio monitoring and device selection
- Configurable voice trigger system
- Robust WebSocket-based client-server architecture

## System Requirements

### Server Requirements
- Python 3.8 or higher
- NVIDIA GPU with at least 4GB VRAM (required for running both Whisper and Kokoro models)
  - GPU acceleration is required for real-time performance
  - CUDA toolkit must be installed for GPU support
- Sufficient disk space for models (approximately 10GB total)

### Client Requirements
- Python 3.8 or higher
- Audio input device (microphone)
- Audio output device (speakers)
- Basic CPU for audio processing

## Installation

### Server Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd audio_chat
```

2. Install server dependencies:
```bash
pip install -r server_requirements.txt
```

3. Set up the required models:
   - Download the Whisper speech-to-text model from [HuggingFace](https://huggingface.co/openai/whisper-large-v3-turbo)
   - Download the Kokoro text-to-speech model from [HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M)
   - Download the desired voice packs for Kokoro from the same repository
   - Create your configuration file (see Configuration section below)
   - Set the downloaded model paths in your config.json

Note: The models are large files (several GB) and require sufficient disk space. Make sure to use the correct paths where you downloaded the models in your config.json file:
```json
"models": {
    "whisper": {
        "path": "/path/to/whisper-large-v3-turbo"
    },
    "kokoro": {
        "path": "/path/to/Kokoro-82M",
        "voice_name": "af"  // Choose your preferred voice pack
    }
}
```

### Client Setup

1. Install client dependencies:
```bash
pip install -r client_requirements.txt
```

## Configuration

The system uses a configuration file to manage all settings. To get started:

1. Copy the default configuration file to create your local config:
```bash
cp default_config.json config.json
```

2. Edit `config.json` with your specific settings:
   - Set paths to your downloaded models
   - Configure server addresses and ports
   - Set API keys
   - Adjust audio processing parameters if needed

Note: `config.json` is ignored by git to keep your local settings and sensitive data private. The `default_config.json` serves as a template with example values.

Here's a detailed explanation of each configuration section:

### Assistant Settings
```json
"assistant": {
    "name": "Mira"  // Trigger word to activate the assistant
}
```

### Server Configuration
```json
"server": {
    "websocket": {
        "host": "10.5.2.10",     // WebSocket server host
        "port": 8765,            // WebSocket server port
        "api_key": "your_key"    // Authentication key for client-server communication
    },
    "gpu_device": "cuda:4",      // GPU device to use for ML models
    "models": {
        "whisper": {
            "path": "/path/to/whisper"  // Path to Whisper model
        },
        "kokoro": {
            "path": "/path/to/kokoro",  // Path to Kokoro model
            "voice_name": "af"          // Voice pack to use
        }
    }
}
```

### LLM Configuration
```json
"llm": {
    "server": "10.5.2.10",           // LLM server host
    "port": 8000,                    // LLM server port
    "model_name": "your_model_name",  // Name of the LLM model to use
    "api_base": "http://10.5.2.11:8000/v1",  // API endpoint
    "api_key": "YOUR_API_KEY_HERE",  // API key for LLM service
    "conversation": {
        "context_timeout": 180,       // Seconds before conversation context expires
        "max_tokens": 8000,          // Maximum tokens for context window
        "temperature": 0.7,          // Response randomness (0.0-1.0)
        "response_max_tokens": 1024   // Maximum tokens per response
    }
}
```

### Audio Processing Settings
```json
"audio_processing": {
    "chunk_size": 2048,          // Audio processing chunk size
    "desired_rate": 16000,       // Target sample rate in Hz
    "time_constants": {
        "peak_attack": 0.001,    // Peak meter attack time (seconds)
        "peak_release": 0.100,   // Peak meter release time (seconds)
        "rms_attack": 0.030,     // RMS meter attack time (seconds)
        "rms_release": 0.500     // RMS meter release time (seconds)
    },
    "noise_floor": {
        "initial": -50.0,        // Initial noise floor (dB)
        "min": -65.0,           // Minimum allowed noise floor (dB)
        "max": -20.0            // Maximum allowed noise floor (dB)
    }
}
```

### Speech Detection Settings
```json
"speech_detection": {
    "preroll_duration": 0.5,     // Audio capture before speech detection (seconds)
    "min_speech_duration": 0.5,  // Minimum duration to consider as speech (seconds)
    "max_silence_duration": 0.8, // Maximum silence before closing capture (seconds)
    "thresholds": {
        "open": 30.0,           // dB above noise floor to start capture
        "close": 25.0           // dB above noise floor to stop capture
    },
    "window_size": 150,         // Analysis window size
    "hold_samples": 15,         // Samples to hold gate open
    "pre_emphasis": 0.97        // Pre-emphasis filter coefficient
}
```

### Client Settings
```json
"client": {
    "retry": {
        "max_attempts": 3,       // Maximum connection retry attempts
        "delay_seconds": 2       // Delay between retry attempts
    }
}
```

## Usage

1. Start the server:
```bash
python server.py
```

2. Start the client:
```bash
python client.py
```

3. The system will:
   - Initialize audio devices
   - Calibrate noise floor
   - Connect to the server
   - Open the audio control interface
   - Begin listening for the trigger word

4. Speak the trigger word (default: "Mira") followed by your query
   - Example: "Mira, what's the weather like today?"
   - The system will process your speech and respond through text-to-speech

## Troubleshooting

- If audio devices aren't detected, check your system's audio settings
- For GPU errors, verify CUDA installation and GPU availability
- Connection issues may require checking firewall settings and network connectivity
- For model loading errors, verify model paths in config.json

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
