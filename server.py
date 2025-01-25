#!/usr/bin/env python3

import asyncio
import websockets
import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time
import sys
import json
sys.path.append('/mnt/models/hexgrad/Kokoro-82M')
from models import build_model
from collections import deque
from src.audio_core import AudioCore
from urllib.parse import urlparse, parse_qs  # Import for URI parsing

# Load configuration
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

# Get trigger word from config (always use lowercase for comparison)
TRIGGER_WORD = CONFIG['assistant']['name'].lower()

################################################################################
# CONFIG & MODEL LOADING
################################################################################

def find_best_gpu():
    """Find the NVIDIA GPU with the most available VRAM (>= 4GB)"""
    if not torch.cuda.is_available():
        return "cpu"
    
    try:
        import subprocess
        import re
        
        # Run nvidia-smi to get memory info
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        best_gpu = None
        max_free_memory = 0
        min_required_gb = 4
        
        # Parse each line of nvidia-smi output
        for line in result.stdout.strip().split('\n'):
            gpu_id, total, used, free = map(int, line.strip().split(', '))
            free_memory_gb = free / 1024  # Convert MiB to GB
            
            print(f"GPU {gpu_id}: {free_memory_gb:.2f}GB free VRAM")
            
            if free_memory_gb >= min_required_gb and free_memory_gb > max_free_memory:
                max_free_memory = free_memory_gb
                best_gpu = gpu_id
        
        if best_gpu is not None:
            return f"cuda:{best_gpu}"
            
    except Exception as e:
        print(f"Error getting GPU memory info: {e}")
        
    return "cpu"

# Set device from config or auto-detect
device = CONFIG['server']['gpu_device']
if device == "auto":
    device = find_best_gpu()
elif not torch.cuda.is_available():
    device = "cpu"

KOKORO_PATH = CONFIG['server']['models']['kokoro']['path']
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

MODEL_PATH = CONFIG['server']['models']['whisper']['path']

print(f"Device set to use {device}")
print("Loading ASR model and processor...")

# Load model and processor
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

# Create the ASR pipeline
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device
)

print("Loading TTS model...")
# Load Kokoro TTS model
tts_model = build_model(f'{KOKORO_PATH}/kokoro-v0_19.pth', device)
VOICE_NAME = CONFIG['server']['models']['kokoro']['voice_name']  # Load voice name from config
tts_voicepack = torch.load(f'{KOKORO_PATH}/voices/{VOICE_NAME}.pt', weights_only=True).to(device)

from kokoro import generate

################################################################################
# AUDIO SERVER
################################################################################

class AudioServer:
    def __init__(self):
        # Load configuration first
        with open('config.json', 'r') as f:
            self.config = json.load(f)
            
        # Initialize audio core with safe defaults
        self.audio_core = AudioCore()
        self.audio_core.noise_floor = -96.0  # Safe default until client connects
        self.audio_core.min_floor = -96.0
        self.audio_core.max_floor = -36.0  # 60dB range
        self.audio_core.rms_level = -96.0
        self.audio_core.peak_level = -96.0
        
        # Voice filtering configuration
        self.enable_voice_filtering = False  # Default to disabled
        
        # Client's calibrated noise floor (will be set when client connects)
        self.client_noise_floor = -96.0  # Start with safe default
        
        # Pre-roll buffer for catching speech starts
        self.preroll_duration = self.config['speech_detection']['preroll_duration']
        self.preroll_samples = int(16000 * self.preroll_duration)  # at 16kHz
        self.preroll_buffer = deque(maxlen=self.preroll_samples)
        self.last_speech_end = 0  # timestamp of last speech end
        
        # Voice profile management (cleared after each speech segment)
        self.current_voice_profile = None
        self.voice_profile_timestamp = None
        
        # Speech detection parameters from config
        self.min_speech_duration = self.config['speech_detection']['min_speech_duration']
        self.max_silence_duration = self.config['speech_detection']['max_silence_duration']
        self.speech_start_time = 0
        self.last_speech_level = -96.0
        
        # Transcript filtering
        self.transcript_history = deque(maxlen=10)  # Increased history
        self.min_confidence = 0.4  # Base confidence threshold
        self.short_phrase_confidence = 0.8  # Higher threshold for short phrases
        self.last_debug_time = 0
        
        # Debounce for repeated transcripts
        self.last_transcript = ""
        self.last_transcript_time = 0
        self.min_repeat_interval = 2.0  # seconds between identical transcripts
        
    def add_to_preroll(self, audio_data):
        """Add audio to pre-roll buffer"""
        for sample in audio_data:
            self.preroll_buffer.append(sample)
            
    def get_preroll_audio(self):
        """Get pre-roll audio if available and appropriate"""
        now = time.time()
        # Only use preroll if we've had sufficient silence
        if now - self.last_speech_end > self.preroll_duration:
            return np.array(self.preroll_buffer)
        return np.array([])
        
    def should_process_transcript(self, transcript, confidence, speech_duration):
        """Determine if transcript should be processed based on sophisticated rules"""
        if not transcript:
            return False
            
        transcript = transcript.strip().lower()
        now = time.time()
        
        # Basic validation
        if len(transcript.strip()) <= 1:
            return False
            
        # Skip common false positives
        if transcript in ["thank you.", "thanks.", "okay.", "ok.", "mm.", "hmm.", "um.", "uh."]:
            return False
            
        # Duration-based validation (ignore confidence since Whisper v3 doesn't provide it)
        if speech_duration < self.min_speech_duration:
            # For very short utterances, be more strict about what we accept
            word_count = len(transcript.split())
            if word_count <= 2:
                # Skip very short phrases unless they contain the trigger word
                if TRIGGER_WORD not in transcript:
                    return False
        
        # Check for repeated transcripts with debounce
        if transcript == self.last_transcript:
            if now - self.last_transcript_time < self.min_repeat_interval:
                return False
            
        # Check recent history (case-insensitive)
        if any(t.lower() == transcript for t in self.transcript_history):
            return False
        
        # Update history and timestamps
        self.transcript_history.append(transcript)
        self.last_transcript = transcript
        self.last_transcript_time = now
        
        return True

class ClientSettingsManager:
    """
    Manages client-specific AudioServer instances.
    """
    def __init__(self):
        self.client_settings = {}

    def get_audio_server(self, client_id):
        """
        Retrieves or creates an AudioServer instance for a given client ID.
        """
        if client_id not in self.client_settings:
            print(f"Creating new AudioServer instance for client ID: {client_id}")
            self.client_settings[client_id] = AudioServer()
        return self.client_settings[client_id]

    def remove_client(self, client_id):
        """
        Removes a client's AudioServer instance when they disconnect.
        """
        if client_id in self.client_settings:
            print(f"Removing AudioServer instance for client ID: {client_id}")
            del self.client_settings[client_id]

# Global ClientSettingsManager instance
client_settings_manager = ClientSettingsManager()


################################################################################
# WEBSOCKET HANDLER
################################################################################

async def process_audio_chunk(websocket, chunk, fade_in=None, fade_out=None, fade_samples=32):
    """Process and send a single audio chunk"""
    try:
        # Apply fades if provided
        if fade_in is not None:
            chunk[:fade_samples] *= fade_in
        if fade_out is not None:
            chunk[-fade_samples:] *= fade_out
            
        # Convert to int16 and send
        chunk_int16 = np.clip(chunk * 32768.0, -32768, 32767).astype(np.int16)
        await websocket.send(b'TTS:' + chunk_int16.tobytes())
    except Exception as e:
        print(f"Error processing audio chunk: {e}")

async def handle_tts(websocket, text, client_id):
    """Handle text-to-speech request and stream audio back to client"""
    try:
        # Get client-specific AudioServer instance
        server = client_settings_manager.get_audio_server(client_id)

        # Start TTS generation immediately
        print(f"\n[TTS] Generating audio for text chunk: {text[:50]}...")
        
        # Create a task for TTS generation
        loop = asyncio.get_event_loop()
        audio_future = loop.run_in_executor(
            None, 
            lambda: generate(tts_model, text, tts_voicepack, lang=VOICE_NAME[0])
        )
        
        # While waiting for TTS generation, prepare processing parameters
        FRAME_SIZE = 512  # Smaller frames for faster initial playback
        fade_samples = 32  # Smaller fade for reduced latency
        fade_in = np.linspace(0, 1, fade_samples).astype(np.float32)
        fade_out = np.linspace(1, 0, fade_samples).astype(np.float32)
        
        # Wait for TTS generation to complete
        audio, _ = await audio_future
        print(f"[TTS] Generated {len(audio)} samples at 24kHz ({len(audio)/24000:.2f} seconds)")
        
        # Convert float audio directly to float32 for processing
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        
        # Process chunks concurrently
        tasks = []
        
        # Send initial chunk with fade in
        if len(audio) >= FRAME_SIZE:
            first_chunk = audio[:FRAME_SIZE].copy()
            tasks.append(process_audio_chunk(websocket, first_chunk, fade_in=fade_in))
        
        # Process remaining chunks
        for i in range(FRAME_SIZE, len(audio) - FRAME_SIZE, FRAME_SIZE):
            chunk = audio[i:i + FRAME_SIZE].copy()
            
            # Add fade out to final chunk
            if i + FRAME_SIZE >= len(audio) - FRAME_SIZE:
                tasks.append(process_audio_chunk(websocket, chunk, fade_out=fade_out))
            else:
                tasks.append(process_audio_chunk(websocket, chunk))
            
            # Process chunks in batches to maintain order while allowing concurrency
            if len(tasks) >= 8:
                await asyncio.gather(*tasks)
                tasks = []
        
        # Process any remaining chunks
        if tasks:
            await asyncio.gather(*tasks)
        
        # Send chunk end marker
        await websocket.send(b'TTS_END')
        
    except Exception as e:
        print(f"TTS Error: {e}")
        await websocket.send("TTS_ERROR")


def verify_api_key(websocket, client_id):
    """Verify the API key and client ID from the websocket connection URI"""
    try:
        # Get server API key from config
        server_api_key = CONFIG['server']['websocket']['api_key']
        if not server_api_key:
            print("No server API key configured")
            return False

        # Verify API Key
        path_string = None
        try: # Try to get path from websocket.request
            path_string = websocket.request.path
            print(f"Path from websocket.request.path: {path_string}")
        except AttributeError:
            print("websocket.request.path not available")
            pass # It's okay if websocket.request.path is not available

        if not path_string: # Fallback to websocket.path if transport method fails
            try:
                path_string = websocket.path
                print(f"Path from websocket.path: {path_string}")
            except AttributeError:
                print("websocket.path also not available")
                return False
        
        # Parse the path to get query parameters
        parsed_uri = urlparse(path_string)
        query_params = parse_qs(parsed_uri.query)

        # Get client API key from query parameters
        client_api_key_list = query_params.get('api_key', [])
        if not client_api_key_list:
            print("No API key provided in URI query parameters")
            return False
        client_api_key = client_api_key_list[0]  # Take the first API key if multiple are present

        # Verify keys match
        if client_api_key != server_api_key:
            print("Client API key does not match server API key")
            return False

        # Verify Client ID
        client_id_list = query_params.get('client_id', [])
        if not client_id_list:
            print("No Client ID provided in URI query parameters")
            return False
        client_id_uri = client_id_list[0]
        if client_id_uri != str(client_id): # Compare string representations
            print(f"Client ID in URI does not match: expected {client_id}, got {client_id_uri}")
            return False
        
        return True

    except Exception as e:
        print(f"Error verifying API key and client ID from URI: {e}")
        return False


async def transcribe_audio(websocket):
    """
    Receives audio chunks in raw PCM form, processes them with professional
    audio techniques, and runs speech recognition when appropriate.
    """
    client_id = None  # Initialize client_id here
    server = None
    audio_buffer = None
    was_speech = False

    try:
        # Extract client ID from URI
        path_string = websocket.request.path if hasattr(websocket.request, 'path') else websocket.path
        parsed_uri = urlparse(path_string)
        query_params = parse_qs(parsed_uri.query)
        client_id_list = query_params.get('client_id', [])
        if not client_id_list:
            print("Client ID missing from URI.")
            await websocket.close(code=4000, reason="Client ID required")  # Close connection with custom code
            return
        client_id = client_id_list[0]

        # Verify API key and client ID
        if not verify_api_key(websocket, client_id):
            print("Client connection rejected: Invalid API key or Client ID")
            await websocket.send("AUTH_FAILED")
            return
        
        # Send authentication success
        await websocket.send("AUTH_OK")
        print(f"Client authenticated. Client ID: {client_id}. Ready to receive audio chunks...")

        # Get client-specific AudioServer instance
        server = client_settings_manager.get_audio_server(client_id)

        async for message in websocket:
            if isinstance(message, bytes):
                try:
                    # Process each chunk individually
                    chunk_data, sr = server.audio_core.bytes_to_float32_audio(message, sample_rate=24000 if message.startswith(b'TTS:') else None)
                    result = server.audio_core.process_audio(chunk_data)
                    
                    # Always update preroll buffer
                    server.add_to_preroll(result['audio'])
                    
                    # Handle speech state changes
                    if result['is_speech'] and not was_speech:
                        # Start new speech segment
                        preroll = server.get_preroll_audio()
                        audio_buffer = bytearray(message)
                        was_speech = True
                        server.speech_start_time = time.time()
                        server.last_speech_level = result['db_level']
                    elif was_speech:
                        if result['is_speech']:
                            # Continue collecting speech
                            audio_buffer.extend(message)
                            server.last_speech_level = max(server.last_speech_level, result['db_level'])
                        else:
                            # Speech ended - wait for complete phrase
                            await asyncio.sleep(server.max_silence_duration)
                            
                            # Multiple final checks to ensure speech has truly ended
                            silence_confirmed = True
                            for _ in range(3):  # Check multiple chunks
                                final_check, _ = server.audio_core.bytes_to_float32_audio(message, sample_rate=24000 if message.startswith(b'TTS:') else None)
                                final_result = server.audio_core.process_audio(final_check)
                                if final_result['is_speech'] or final_result['db_level'] > server.last_speech_level - 3:
                                    silence_confirmed = False
                                    break
                                await asyncio.sleep(0.05)  # Short delay between checks
                            
                            if silence_confirmed:
                                if audio_buffer:
                                    # Convert buffer to audio
                                    audio_data, sr = server.audio_core.bytes_to_float32_audio(audio_buffer, sample_rate=24000 if audio_buffer.startswith(b'TTS:') else None)
                                    
                                    # Add preroll if available
                                    preroll = server.get_preroll_audio()
                                    if len(preroll) > 0:
                                        audio_data = np.concatenate([preroll, audio_data])
                                    
                                    # Calculate speech duration
                                    speech_duration = time.time() - server.speech_start_time
                                    
                                    # Run speech recognition
                                    asr_result = asr_pipeline(
                                        {"array": audio_data, "sampling_rate": sr},
                                        return_timestamps=True,
                                        generate_kwargs={
                                            "task": "transcribe",
                                            "language": "english",
                                            "use_cache": False
                                        }
                                    )
                                    
                                    # Update last speech end time for preroll management
                                    server.last_speech_end = time.time()
                                    
                                    # Get transcript and confidence
                                    transcript = asr_result["text"].strip()
                                    confidence = asr_result.get("confidence", 0.0)
                                    
                                    # Process transcript with improved filtering
                                    if server.should_process_transcript(transcript, confidence, speech_duration):
                                        try:
                                            if isinstance(transcript, bytes):
                                                transcript = transcript.decode('utf-8')
                                            transcript_str = str(transcript)
                                            print(f"\nTranscript: '{transcript_str}' (confidence: {confidence:.2f}, duration: {speech_duration:.2f}s)")
                                            await websocket.send(transcript_str)
                                        except Exception as e:
                                            print(f"Error processing transcript: {e}")
                                    else:
                                        print(f"\nFiltered transcript: '{transcript}' (confidence: {confidence:.2f}, duration: {speech_duration:.2f}s)")
                            
                            # Reset for next speech segment
                            audio_buffer = None
                            was_speech = False
                            # Clear voice profile at end of speech segment
                            server.current_voice_profile = None
                            server.voice_profile_timestamp = None
                    
                except Exception as e:
                    print(f"Error processing audio: {e}")
                    audio_buffer = bytearray()
                    was_speech = False

            elif isinstance(message, str):
                if message.startswith("NOISE_FLOOR:"):
                    try:
                        parts = message.split(":")
                        noise_floor = float(parts[1])
                        message_client_id = parts[2] if len(parts) > 2 else None

                        if message_client_id != client_id:
                            print(f"Warning: Client ID mismatch in NOISE_FLOOR message. Expected {client_id}, got {message_client_id}")
                            continue # Skip processing if client IDs don't match

                        if noise_floor > -120 and noise_floor < 0:  # Validate reasonable range
                            server.client_noise_floor = float(noise_floor)
                            print(f"\nReceived client noise floor for client ID {client_id}: {noise_floor:.1f} dB")
                            
                            try:
                                # Update AudioCore's noise floor with explicit float conversions
                                server.audio_core.noise_floor = float(noise_floor)
                                server.audio_core.min_floor = float(noise_floor - 5)  # Allow some variation below
                                server.audio_core.max_floor = float(noise_floor + 45)  # Allow speech to be well above
                                
                                print("Updated server speech detection thresholds")
                                
                                # Send ready response to client
                                await websocket.send("READY")
                                print("Sent ready response to client")
                                
                            except Exception as e:
                                print(f"Error updating audio levels: {e}")
                                await websocket.send("ERROR:Failed to update audio levels")
                        else:
                            print(f"\nWarning: Received invalid noise floor value: {noise_floor} dB")
                            await websocket.send("ERROR:Invalid noise floor value")
                    except Exception as e:
                        print(f"Error processing noise floor message: {e}")
                        await websocket.send("ERROR:Failed to process noise floor")
                elif message.strip() == "VOICE_FILTER_ON":
                    server.enable_voice_filtering = True
                    print("\nVoice filtering enabled for client ID {client_id}")
                elif message.strip() == "VOICE_FILTER_OFF":
                    server.enable_voice_filtering = False
                    server.current_voice_profile = None
                    server.voice_profile_timestamp = None
                    print("\nVoice filtering disabled for client ID {client_id}")
                elif message.strip() == "RESET":
                    audio_buffer = bytearray()
                    was_speech = False
                    print(f"Buffer has been reset by client request for client ID {client_id}.")
                elif message.strip() == "EXIT":
                    print(f"Client requested exit. Closing connection for client ID {client_id}.")
                    break
                elif message.startswith("TTS:"):
                    # Handle TTS request asynchronously
                    text = message[4:].strip()  # Remove TTS: prefix
                    print(f"TTS Request for client ID {client_id}: {text}")
                    # Create task for TTS processing to run concurrently
                    asyncio.create_task(handle_tts(websocket, text, client_id))
                else:
                    print(f"Received unknown text message from client ID {client_id}: {message}")

    except websockets.ConnectionClosed as e:
        print(f"Client disconnected. Client ID: {client_id}, Reason: {e}")
    except Exception as e:
        print(f"Server error for client ID {client_id}: {e}")
    finally:
        if server and server.audio_core:
            server.audio_core.close()
        if client_id:
            client_settings_manager.remove_client(client_id)
        if audio_buffer:
            audio_buffer.clear()
        print(f"Cleaned up server resources for client ID {client_id}")


################################################################################
# MAIN SERVER ENTRY POINT
################################################################################

async def main():
    try:
        # Start the server using config
        host = "0.0.0.0"  # Always bind to all interfaces
        port = CONFIG['server']['websocket']['port']
        async with websockets.serve(transcribe_audio, host, port):
            print(f"WebSocket server started on ws://{host}:{port}")
            await asyncio.Future()  # keep running
    except Exception as e:
        print(f"Server startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
