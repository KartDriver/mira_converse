#!/usr/bin/env python3

import asyncio
import websockets
import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time
import sys
import json
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
        
        # Run nvidia-smi to get memory info
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
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
sys.path.append(KOKORO_PATH)
from models import build_model
tts_model = build_model(f'{KOKORO_PATH}/kokoro-v0_19.pth', device)
VOICE_NAME = CONFIG['server']['models']['kokoro']['voice_name']
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
        self.client_noise_floor = -96.0
        
        # Pre-roll buffer for catching speech starts
        self.preroll_duration = self.config['speech_detection']['preroll_duration']
        self.preroll_samples = int(16000 * self.preroll_duration)  # at 16kHz
        self.preroll_buffer = deque(maxlen=self.preroll_samples)
        self.last_speech_end = 0  # timestamp of last speech end
        
        # Voice profile management
        self.current_voice_profile = None
        self.voice_profile_timestamp = None
        
        # Speech detection parameters
        self.min_speech_duration = self.config['speech_detection']['min_speech_duration']
        self.end_silence_duration = self.config['speech_detection']['end_silence_duration']
        self.speech_start_time = 0
        self.last_speech_level = -96.0
        
        # Transcript filtering
        self.transcript_history = deque(maxlen=10)
        self.min_confidence = 0.4  # Base confidence threshold
        self.short_phrase_confidence = 0.8  # Higher threshold for short phrases
        self.last_debug_time = 0
        
        # Debounce for repeated transcripts
        self.last_transcript = ""
        self.last_transcript_time = 0
        self.min_repeat_interval = 2.0  # seconds between identical transcripts

    def add_to_preroll(self, audio_data):
        """Add audio to pre-roll buffer (used to capture the 'start' of speech)."""
        for sample in audio_data:
            self.preroll_buffer.append(sample)
            
    def get_preroll_audio(self):
        """Get pre-roll audio if we've had enough silence to reset."""
        now = time.time()
        if now - self.last_speech_end > self.preroll_duration:
            return np.array(self.preroll_buffer)
        return np.array([])

    def should_process_transcript(self, transcript, confidence, speech_duration):
        """
        Decide if the server should accept the transcript. This helps reject
        false positives, repeated transcripts, very short nonsense, etc.
        """
        if not transcript:
            return False
            
        transcript = transcript.strip().lower()
        now = time.time()
        
        # Basic validations
        if len(transcript) <= 1:
            return False

        # Skip certain common short filler words
        if transcript in ["thank you.", "thanks.", "okay.", "ok.", "mm.", "hmm.", "um.", "uh."]:
            return False

        # For extremely short utterances, be more strict
        if speech_duration < self.min_speech_duration:
            word_count = len(transcript.split())
            if word_count <= 2 and TRIGGER_WORD not in transcript:
                return False

        # Debounce identical transcripts
        if transcript == self.last_transcript:
            if now - self.last_transcript_time < self.min_repeat_interval:
                return False

        # Check recent history for duplicates
        if any(t.lower() == transcript for t in self.transcript_history):
            return False

        # Passed all filters
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
# TTS HELPER FUNCTIONS
################################################################################

async def process_audio_chunk(websocket, chunk, fade_in=None, fade_out=None, fade_samples=32):
    """Send a single chunk of TTS audio to the client, applying fades if needed."""
    try:
        if fade_in is not None:
            chunk[:fade_samples] *= fade_in
        if fade_out is not None:
            chunk[-fade_samples:] *= fade_out
            
        chunk_int16 = np.clip(chunk * 32768.0, -32768, 32767).astype(np.int16)
        await websocket.send(b'TTS:' + chunk_int16.tobytes())
    except Exception as e:
        print(f"Error sending TTS audio chunk: {e}")

async def handle_tts(websocket, text, client_id):
    """
    Handle text-to-speech request and stream audio chunks back to the client.
    Uses the Kokoro TTS model in streaming fashion for lower latency.
    """
    try:
        server = client_settings_manager.get_audio_server(client_id)
        print(f"\n[TTS] Generating audio for text chunk (client {client_id}): '{text}'")

        loop = asyncio.get_event_loop()
        audio_future = loop.run_in_executor(
            None,
            lambda: generate(tts_model, text, tts_voicepack, lang=VOICE_NAME[0])
        )
        
        audio, _ = await audio_future
        print(f"[TTS] Generated {len(audio)} samples ({len(audio)/24000:.2f}s at 24kHz)")

        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

        FRAME_SIZE = 512
        fade_samples = 32
        fade_in = np.linspace(0, 1, fade_samples).astype(np.float32)
        fade_out = np.linspace(1, 0, fade_samples).astype(np.float32)

        tasks = []
        
        # Send initial chunk with fade in
        if len(audio) >= FRAME_SIZE:
            first_chunk = audio[:FRAME_SIZE].copy()
            tasks.append(process_audio_chunk(websocket, first_chunk, fade_in=fade_in))

        # Process the rest
        for i in range(FRAME_SIZE, len(audio) - FRAME_SIZE, FRAME_SIZE):
            chunk = audio[i:i + FRAME_SIZE].copy()
            # Fade out on the last chunk
            if i + FRAME_SIZE >= len(audio) - FRAME_SIZE:
                tasks.append(process_audio_chunk(websocket, chunk, fade_out=fade_out))
            else:
                tasks.append(process_audio_chunk(websocket, chunk))

            # Process chunks in batches to ensure order
            if len(tasks) >= 8:
                await asyncio.gather(*tasks)
                tasks = []

        # Process any remaining
        if tasks:
            await asyncio.gather(*tasks)

        await websocket.send(b'TTS_END')
        
    except Exception as e:
        print(f"TTS Error: {e}")
        await websocket.send("TTS_ERROR")

################################################################################
# SECURITY CHECK
################################################################################

def verify_api_key(websocket, client_id):
    """Verify the API key and client ID from the websocket connection URI."""
    try:
        server_api_key = CONFIG['server']['websocket']['api_key']
        if not server_api_key:
            print("No server API key configured")
            return False

        path_string = None
        try:
            path_string = websocket.request.path
            print(f"Path from websocket.request.path: {path_string}")
        except AttributeError:
            print("websocket.request.path not available")

        if not path_string:
            try:
                path_string = websocket.path
                print(f"Path from websocket.path: {path_string}")
            except AttributeError:
                print("websocket.path also not available")
                return False

        parsed_uri = urlparse(path_string)
        query_params = parse_qs(parsed_uri.query)

        # Client API key
        client_api_key_list = query_params.get('api_key', [])
        if not client_api_key_list:
            print("No API key provided in URI query parameters")
            return False
        client_api_key = client_api_key_list[0]

        # Compare
        if client_api_key != server_api_key:
            print("Client API key does not match server API key")
            return False

        # Verify Client ID
        client_id_list = query_params.get('client_id', [])
        if not client_id_list:
            print("No Client ID provided in URI query parameters")
            return False
        client_id_uri = client_id_list[0]
        if client_id_uri != str(client_id):
            print(f"Client ID mismatch: expected {client_id}, got {client_id_uri}")
            return False
        
        return True

    except Exception as e:
        print(f"Error verifying API key and client ID from URI: {e}")
        return False

################################################################################
# MAIN ASR HANDLER
################################################################################

async def transcribe_audio(websocket):
    """
    Receives audio chunks from the client, processes them in real-time (using
    AudioCore's gating and calibration logic), and when speech is detected,
    sends transcripts back to the client.
    """
    client_id = None
    server = None
    audio_buffer = None
    was_speech = False

    try:
        # Extract client ID
        path_string = websocket.request.path if hasattr(websocket.request, 'path') else websocket.path
        parsed_uri = urlparse(path_string)
        query_params = parse_qs(parsed_uri.query)
        client_id_list = query_params.get('client_id', [])
        if not client_id_list:
            print("Client ID missing from URI.")
            await websocket.close(code=4000, reason="Client ID required")
            return
        client_id = client_id_list[0]

        # Verify API
        if not verify_api_key(websocket, client_id):
            print("Client connection rejected: Invalid API key or Client ID")
            await websocket.send("AUTH_FAILED")
            return
        
        # Auth success
        await websocket.send("AUTH_OK")
        print(f"Client authenticated. Client ID: {client_id}.")

        # Get or create AudioServer for this client
        server = client_settings_manager.get_audio_server(client_id)

        async for message in websocket:
            if isinstance(message, bytes):
                # Audio data
                try:
                    # Convert to float32
                    chunk_data, sr = server.audio_core.bytes_to_float32_audio(message, sample_rate=16000)
                    
                    # Skip multi-band calibration if we suspect speech to avoid polluting noise floor
                    # We let process_audio handle gating logic:
                    result = server.audio_core.process_audio(chunk_data)

                    # Pre-roll always accumulates the raw audio
                    server.add_to_preroll(result['audio'])

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
                            # Speech ended (audio_core has already waited for required silence)
                            if not result['is_speech']:
                                # Confirm speech ended
                                if audio_buffer:
                                    # Combine with preroll
                                    audio_data, sr = server.audio_core.bytes_to_float32_audio(audio_buffer, sample_rate=16000)
                                    preroll = server.get_preroll_audio()
                                    if len(preroll) > 0:
                                        audio_data = np.concatenate([preroll, audio_data])

                                    speech_duration = time.time() - server.speech_start_time

                                    # Run ASR
                                    asr_result = asr_pipeline(
                                        {"array": audio_data, "sampling_rate": sr},
                                        return_timestamps=True,
                                        generate_kwargs={
                                            "task": "transcribe",
                                            "language": "english",
                                            "use_cache": False
                                        }
                                    )

                                    server.last_speech_end = time.time()
                                    transcript = asr_result["text"].strip()
                                    confidence = asr_result.get("confidence", 0.0)

                                    if server.should_process_transcript(transcript, confidence, speech_duration):
                                        try:
                                            if isinstance(transcript, bytes):
                                                transcript = transcript.decode('utf-8')
                                            transcript_str = str(transcript)
                                            print(f"\nTranscript: '{transcript_str}' (client: {client_id})")
                                            await websocket.send(transcript_str)
                                        except Exception as e:
                                            print(f"Error sending transcript: {e}")
                                    else:
                                        print(f"\nFiltered transcript: '{transcript}' (client: {client_id})")

                                # Reset states
                                audio_buffer = None
                                was_speech = False
                                server.current_voice_profile = None
                                server.voice_profile_timestamp = None
                            else:
                                # Possibly still speech - re-append chunk
                                audio_buffer.extend(message)

                except Exception as e:
                    print(f"Error processing audio chunk from client {client_id}: {e}")
                    audio_buffer = bytearray()
                    was_speech = False

            elif isinstance(message, str):
                # Control messages
                if message.startswith("NOISE_FLOOR:"):
                    """
                    The client sends an initial, measured noise floor.
                    We update the server's AudioCore to align the thresholds.
                    """
                    try:
                        parts = message.split(":")
                        noise_floor = float(parts[1])
                        message_client_id = parts[2] if len(parts) > 2 else None

                        if message_client_id != client_id:
                            print(f"Warning: Client ID mismatch in NOISE_FLOOR. Expected {client_id}, got {message_client_id}")
                            continue

                        if -120 < noise_floor < 0:
                            server.client_noise_floor = noise_floor
                            server.audio_core.noise_floor = float(noise_floor)
                            server.audio_core.min_floor = float(noise_floor - 5)
                            server.audio_core.max_floor = float(noise_floor + 45)

                            print(f"\nClient {client_id} noise floor set to: {noise_floor:.1f} dB")
                            await websocket.send("READY")

                        else:
                            print(f"Invalid noise floor from client {client_id}: {noise_floor}")
                            await websocket.send("ERROR:Invalid noise floor value")

                    except Exception as e:
                        print(f"Error processing noise floor message from {client_id}: {e}")
                        await websocket.send("ERROR:Failed to process noise floor")

                elif message.strip() == "VOICE_FILTER_ON":
                    server.enable_voice_filtering = True
                    print(f"\nVoice filtering enabled for client {client_id}")
                elif message.strip() == "VOICE_FILTER_OFF":
                    server.enable_voice_filtering = False
                    server.current_voice_profile = None
                    server.voice_profile_timestamp = None
                    print(f"\nVoice filtering disabled for client {client_id}")
                elif message.strip() == "RESET":
                    audio_buffer = bytearray()
                    was_speech = False
                    print(f"Buffer reset by client {client_id}.")
                elif message.strip() == "EXIT":
                    print(f"Client {client_id} requested exit. Closing connection.")
                    break
                elif message.startswith("TTS:"):
                    text = message[4:].strip()
                    print(f"TTS request from client {client_id}: {text}")
                    asyncio.create_task(handle_tts(websocket, text, client_id))
                else:
                    print(f"Unknown message from client {client_id}: {message}")

    except websockets.ConnectionClosed as e:
        print(f"Client {client_id} disconnected: {e}")
    except Exception as e:
        print(f"Server error for client {client_id}: {e}")
    finally:
        if server and server.audio_core:
            server.audio_core.close()
        if client_id:
            client_settings_manager.remove_client(client_id)
        if audio_buffer:
            audio_buffer.clear()
        print(f"Cleaned up server resources for client {client_id}")

################################################################################
# MAIN
################################################################################

async def main():
    try:
        host = "0.0.0.0"
        port = CONFIG['server']['websocket']['port']
        async with websockets.serve(transcribe_audio, host, port):
            print(f"WebSocket server started on ws://{host}:{port}")
            await asyncio.Future()  # keep running
    except Exception as e:
        print(f"Server startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
