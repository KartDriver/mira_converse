#!/usr/bin/env python3

import asyncio
import websockets
import torch
import io
import soundfile as sf
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

    # For demonstration, define a threshold for ~2 seconds at 16kHz, 16-bit, mono:
    # 16,000 samples/sec * 2 bytes/sample * 2 sec = 64,000 bytes
    CHUNK_THRESHOLD = 16000

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Append raw PCM bytes from the client
                audio_buffer.extend(message)
                print(f"Received {len(message)} bytes. Buffer size = {len(audio_buffer)} bytes.")

                # If buffer is "big enough" for a chunk, let's transcribe
                if len(audio_buffer) >= CHUNK_THRESHOLD:
                    try:
                        # Decode raw PCM to float32
                        audio_data, sr = bytes_to_float32_audio(audio_buffer)

                        # Pass audio to the pipeline via dict format (to specify sampling rate)
                        audio_input = {"array": audio_data, "sampling_rate": sr}
                        # audio_data = bytes(audio_buffer)
                        result = asr_pipeline(audio_input, generate_kwargs={"language": "english"}, return_timestamps=True)
                        transcript = result["text"]
                        print(f"Transcript: {transcript}")
                    except Exception as e:
                        transcript = f"[Error transcribing audio: {e}]"
                        print(transcript)

                    # Send transcript back to the client
                    await websocket.send(transcript)

                    # Clear the buffer
                    audio_buffer.clear()

            else:
                # Non-binary messages are control signals from the client
                text_msg = message.strip()
                if text_msg == "RESET":
                    audio_buffer.clear()
                    print("Buffer has been reset by client request.")
                elif text_msg == "EXIT":
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
