#!/usr/bin/env python3

import asyncio
import websockets
import torch
import io
import numpy as np
import soundfile as sf
from scipy import signal
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

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
# AUDIO DECODING UTILITY
################################################################################

def bytes_to_float32_audio(raw_bytes):
    """
    Convert raw bytes to float32 audio data using soundfile.
    """
    audio_buffer = io.BytesIO(raw_bytes)
    audio_data, sample_rate = sf.read(audio_buffer, dtype='float32', format='RAW', subtype='PCM_16', samplerate=16000, channels=1)
    return audio_data, sample_rate

################################################################################
# WEBSOCKET HANDLER
################################################################################

async def transcribe_audio(websocket):
    """
    Receives audio chunks in raw PCM form,
    accumulates them until we have a "big enough" buffer,
    runs inference, sends back transcript, and then clears the buffer.
    """
    print("Client connected. Ready to receive audio chunks...")
    audio_buffer = bytearray()
    # Speech detection settings
    speech_threshold = -50.0  # Default threshold, will be updated by client
    is_gate_open = False
    hold_counter = 0
    hold_samples = 5  # Hold gate open for 5 frames
    
    # Pre-emphasis filter
    pre_emphasis = 0.97
    prev_sample = 0.0
    
    # Buffer settings (16-bit audio at 16kHz)
    MIN_CHUNK_SIZE = 32000  # Minimum size to process (~1 second)
    MAX_CHUNK_SIZE = 64000  # Maximum size to accumulate (~2 seconds)

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Append raw PCM bytes from the client
                audio_buffer.extend(message)
                print(f"Received {len(message)} bytes. Buffer size = {len(audio_buffer)} bytes.")

                # Process if we have enough data or buffer is getting too large
                current_size = len(audio_buffer)
                if current_size >= MIN_CHUNK_SIZE or current_size >= MAX_CHUNK_SIZE:
                    should_clear_buffer = True  # Default to clearing buffer
                    
                    try:
                        # Decode raw PCM to float32
                        audio_data, sr = bytes_to_float32_audio(audio_buffer)

                        # Check audio characteristics
                        rms = np.sqrt(np.mean(audio_data**2))
                        db = 20 * np.log10(rms) if rms > 0 else -96
                        # Improved zero-crossing calculation with DC offset removal
                        dc_removed = audio_data - np.mean(audio_data)  # Remove DC offset
                        zero_crossings = np.sum(np.abs(np.diff(np.signbit(dc_removed)))) / len(dc_removed)
                        
                        # Log audio characteristics
                        print(f"\nAudio Analysis:")
                        print(f"- Buffer size: {len(audio_buffer)} bytes")
                        print(f"- Audio samples: {len(audio_data)}")
                        print(f"- RMS: {rms:.6f}")
                        print(f"- dB: {db:.1f}")
                        print(f"- Zero-crossing rate: {zero_crossings:.6f}")
                        
                        # Apply pre-emphasis filter
                        emphasized = np.zeros_like(audio_data)
                        emphasized[0] = audio_data[0] - pre_emphasis * prev_sample
                        emphasized[1:] = audio_data[1:] - pre_emphasis * audio_data[:-1]
                        prev_sample = audio_data[-1]
                        
                        # Calculate spectral flatness for speech/noise discrimination
                        _, psd = signal.welch(emphasized, fs=16000, nperseg=256)
                        spectral_flatness = np.exp(np.mean(np.log(psd + 1e-10))) / np.mean(psd + 1e-10)
                        is_speech_like = spectral_flatness < 0.3
                        
                        # Professional speech detection with hysteresis
                        if not is_gate_open:
                            # Check if should open gate
                            if db > speech_threshold and zero_crossings > 0.0003 and is_speech_like:  # Lower zero-crossing threshold
                                is_gate_open = True
                                hold_counter = hold_samples
                        else:
                            # Check if should close gate
                            if db < speech_threshold - 0.2:  # Add small hysteresis
                                if hold_counter > 0:
                                    hold_counter -= 1
                                else:
                                    is_gate_open = False
                            else:
                                hold_counter = hold_samples
                        
                        # Basic speech detection
                        if is_gate_open:
                            # Use pre-emphasized audio for better speech recognition
                            audio_input = {"array": emphasized, "sampling_rate": sr}
                            result = asr_pipeline(audio_input, generate_kwargs={"language": "english"}, return_timestamps=True)
                            transcript = result["text"].strip()
                            
                            # Process audio and filter transcripts
                            if transcript and len(transcript.strip()) > 0:
                                transcript_lower = transcript.strip().lower()
                                # Skip very short or common filler phrases
                                skip_phrases = {"thank you", "thanks", "thank", "okay", "ok", 
                                             "mm", "hmm", "uh", "um", "ah", "oh", "i", "a"}
                                if (len(transcript_lower) > 1 and 
                                    not any(phrase == transcript_lower for phrase in skip_phrases)):
                                    print(f"- Transcript: '{transcript}'")
                                    await websocket.send(transcript)
                                else:
                                    print(f"- Skipping filler: '{transcript}'")
                        else:
                            print(f"Skipping processing - Gate closed")
                            # Only clear if we've accumulated too much data
                            should_clear_buffer = current_size >= MAX_CHUNK_SIZE
                            
                    except Exception as e:
                        print(f"Error processing audio: {e}")
                        should_clear_buffer = True  # Clear buffer on error
                    
                    # Clear buffer if needed
                    if should_clear_buffer:
                        audio_buffer.clear()
                        print("Buffer cleared")

            elif isinstance(message, str):
                if message.startswith("SPEECH_THRESHOLD:"):
                    try:
                        # Update speech threshold from client calibration
                        speech_threshold = float(message.split(":")[1])
                        print(f"Speech threshold updated to: {speech_threshold:.1f} dB")
                    except ValueError as e:
                        print(f"Error parsing noise floor value: {e}")
                elif message.strip() == "RESET":
                    audio_buffer.clear()
                    print("Buffer has been reset by client request.")
                elif message.strip() == "EXIT":
                    print("Client requested exit. Closing connection.")
                    break
                else:
                    print(f"Received unknown text message: {text_msg}")

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
