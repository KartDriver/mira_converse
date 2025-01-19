#!/usr/bin/env python3

import asyncio
import websockets
import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time
from collections import deque
from audio_core import AudioCore

################################################################################
# CONFIG & MODEL LOADING
################################################################################

# Change "cuda:4" to your preferred GPU index or use "cuda:0" if you only have one GPU
device = "cuda:4" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

MODEL_PATH = "/mnt/models/openai/whisper-large-v3-turbo"

print(f"Device set to use {device}")
print("Loading model and processor...")

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,  # or False if you don't have safetensors
).to(device)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

# Create the ASR pipeline
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

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

async def transcribe_audio(websocket):
    """
    Receives audio chunks in raw PCM form, processes them with professional
    audio techniques, and runs speech recognition when appropriate.
    """
    print("Client connected. Ready to receive audio chunks...")
    audio_buffer = bytearray()
    server = AudioServer()
    
    # Buffer settings (16-bit audio at 16kHz)
    MIN_CHUNK_SIZE = 32000  # Minimum size to process (~1 second)
    MAX_CHUNK_SIZE = 64000  # Maximum size to accumulate (~2 seconds)

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Append raw PCM bytes from the client
                audio_buffer.extend(message)
                
                # Process if we have enough data or buffer is getting too large
                current_size = len(audio_buffer)
                if current_size >= MIN_CHUNK_SIZE or current_size >= MAX_CHUNK_SIZE:
                    should_clear_buffer = True  # Default to clearing buffer
                    
                    try:
                        # Process audio with professional techniques
                        audio_data, sr = server.audio_core.bytes_to_float32_audio(audio_buffer)
                        result = server.audio_core.process_audio(audio_data)
                        
                        # Process speech if detected
                        if result['is_speech']:
                            # Get pre-roll audio if appropriate
                            preroll_audio = server.get_preroll_audio()
                            
                            # Combine pre-roll with current audio if available
                            if len(preroll_audio) > 0:
                                combined_audio = np.concatenate([preroll_audio, result['audio']])
                                print(f"Added {len(preroll_audio)/16000:.3f}s of pre-roll audio")
                            else:
                                combined_audio = result['audio']
                            
                            # Log detailed analysis when speech is detected
                            print(f"\nSpeech Detected:")
                            print(f"- dB Level: {result['db_level']:.1f}")
                            print(f"- Noise Floor: {result['noise_floor']:.1f}")
                            print(f"- Speech Ratio: {result['speech_ratio']:.3f}")
                            print(f"- Zero-crossing rate: {result['zero_crossings']:.6f}")
                            
                            # Use combined audio for speech recognition
                            audio_input = {"array": combined_audio, "sampling_rate": sr}
                            asr_result = asr_pipeline(
                                audio_input, 
                                generate_kwargs={"language": "english", "condition_on_prev_tokens": True}, 
                                return_timestamps=True
                            )
                            
                            # Get transcript and confidence
                            transcript = asr_result["text"].strip()
                            confidence = asr_result.get("confidence", 0.0)
                            
                            # Skip "Thank you." responses
                            if transcript.lower().strip() != "thank you.":
                                print(f"- Transcript: '{transcript}'")
                                await websocket.send(transcript)
                        else:
                            # Simple status indicator for non-speech
                            if current_size >= MAX_CHUNK_SIZE:
                                print(".", end="", flush=True)  # Progress indicator
                            # Only clear if we've accumulated too much data
                            should_clear_buffer = current_size >= MAX_CHUNK_SIZE
                            
                    except Exception as e:
                        print(f"Error processing audio: {e}")
                        should_clear_buffer = True  # Clear buffer on error
                    
                    # Clear buffer if needed
                    if should_clear_buffer:
                        audio_buffer.clear()
                        if result['is_speech']:  # Only print buffer clear during speech
                            print("\nBuffer cleared")

            elif isinstance(message, str):
                if message.strip() == "RESET":
                    audio_buffer.clear()
                    print("Buffer has been reset by client request.")
                elif message.strip() == "EXIT":
                    print("Client requested exit. Closing connection.")
                    break
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
