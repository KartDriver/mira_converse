#!/usr/bin/env python3

import asyncio
import websockets
import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time
import sys
sys.path.append('/mnt/models/hexgrad/Kokoro-82M')
from models import build_model
#!/usr/bin/env python3

import asyncio
import websockets
from collections import deque
from audio_core import AudioCore

################################################################################
# CONFIG & MODEL LOADING
################################################################################

# Change "cuda:4" to your preferred GPU index or use "cuda:0" if you only have one GPU
device = "cuda:4" if torch.cuda.is_available() else "cpu"
KOKORO_PATH = "/mnt/models/hexgrad/Kokoro-82M"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

MODEL_PATH = "/mnt/models/openai/whisper-large-v3-turbo"

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
VOICE_NAME = 'af'  # Default voice (50-50 mix of Bella & Sarah)
tts_voicepack = torch.load(f'{KOKORO_PATH}/voices/{VOICE_NAME}.pt', weights_only=True).to(device)

from kokoro import generate

################################################################################
# AUDIO SERVER
################################################################################

class AudioServer:
    def __init__(self):
        self.audio_core = AudioCore()
        # Pre-roll buffer for catching speech starts
        self.preroll_duration = 0.3  # seconds
        self.preroll_samples = int(16000 * self.preroll_duration)  # at 16kHz
        self.preroll_buffer = deque(maxlen=self.preroll_samples)
        self.last_speech_end = 0  # timestamp of last speech end
        # Transcript filtering
        self.transcript_history = deque(maxlen=5)
        self.min_speech_duration = 0.3  # seconds
        self.last_debug_time = 0
        
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

    def should_process_transcript(self, transcript, confidence):
        """Determine if transcript should be processed based on sophisticated rules"""
        if not transcript or confidence < 0.4:
            return False
            
        transcript = transcript.strip().lower()
        
        # Skip "Thank you." response (case insensitive)
        if transcript == "thank you.":
            return False
            
        # Skip empty or single character transcripts
        if len(transcript.strip()) <= 1:
            return False
            
        # Skip if it's a repeat of recent transcript
        if transcript in self.transcript_history:
            return False
            
        # Skip very short phrases unless they have high confidence
        if len(transcript.split()) <= 2 and confidence < 0.8:
            return False
            
        # Add to history and return True if passed all filters
        self.transcript_history.append(transcript)
        return True

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

async def handle_tts(websocket, text):
    """Handle text-to-speech request and stream audio back to client"""
    try:
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

async def transcribe_audio(websocket):
    """
    Receives audio chunks in raw PCM form, processes them with professional
    audio techniques, and runs speech recognition when appropriate.
    """
    print("Client connected. Ready to receive audio chunks...")
    server = AudioServer()
    audio_buffer = None
    was_speech = False

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                try:
                    # Process each chunk individually
                    chunk_data, sr = server.audio_core.bytes_to_float32_audio(message)
                    result = server.audio_core.process_audio(chunk_data)
                    
                    # Always update preroll buffer
                    server.add_to_preroll(result['audio'])
                    
                    # Handle speech state changes
                    if result['is_speech'] and not was_speech:
                        # Start new speech segment
                        preroll = server.get_preroll_audio()
                        audio_buffer = bytearray(message)
                        was_speech = True
                    elif was_speech:
                        if result['is_speech']:
                            # Continue collecting speech
                            audio_buffer.extend(message)
                        else:
                            # Speech ended - wait a short moment to ensure we have the complete phrase
                            await asyncio.sleep(0.1)  # 100ms delay
                            
                            # Final check of the chunk to confirm speech has ended
                            final_check, _ = server.audio_core.bytes_to_float32_audio(message)
                            final_result = server.audio_core.process_audio(final_check)
                            
                            if not final_result['is_speech']:
                                if audio_buffer:
                                    # Convert buffer to audio
                                    audio_data, sr = server.audio_core.bytes_to_float32_audio(audio_buffer)
                                    
                                    # Add preroll if available
                                    preroll = server.get_preroll_audio()
                                    if len(preroll) > 0:
                                        audio_data = np.concatenate([preroll, audio_data])
                                    
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
                                    
                                    # Skip "Thank you." responses
                                    if transcript.lower().strip() != "thank you.":
                                        print(f"- Transcript: '{transcript}'")
                                        await websocket.send(transcript)
                            
                            # Reset for next speech segment
                            audio_buffer = None
                            was_speech = False
                    
                except Exception as e:
                    print(f"Error processing audio: {e}")
                    audio_buffer = bytearray()
                    was_speech = False

            elif isinstance(message, str):
                if message.strip() == "RESET":
                    audio_buffer = bytearray()
                    was_speech = False
                    print("Buffer has been reset by client request.")
                elif message.strip() == "EXIT":
                    print("Client requested exit. Closing connection.")
                    break
                elif message.startswith("TTS:"):
                    # Handle TTS request asynchronously
                    text = message[4:].strip()  # Remove TTS: prefix
                    print(f"TTS Request: {text}")
                    # Create task for TTS processing to run concurrently
                    asyncio.create_task(handle_tts(websocket, text))
                else:
                    print(f"Received unknown text message: {message}")

    except websockets.ConnectionClosed as e:
        print(f"Client disconnected. Reason: {e}")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        print("Exiting transcribe loop.")

################################################################################
# MAIN SERVER ENTRY POINT
################################################################################

async def main():
    # Start the server on port 8765
    async with websockets.serve(transcribe_audio, "0.0.0.0", 8765):
        print("WebSocket server started on ws://0.0.0.0:8765")
        await asyncio.Future()  # keep running

if __name__ == "__main__":
    asyncio.run(main())
