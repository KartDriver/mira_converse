#!/usr/bin/env python3
"""
Audio Chat Client

This script captures audio from your microphone, sends it to a server for real-time
transcription, and displays the transcribed text. It automatically selects the most
appropriate input device (preferring built-in microphone on MacBooks) and handles
audio capture and streaming.

Usage:
    python client.py

The script will automatically:
1. Connect to the transcription server
2. Select the best available microphone (built-in mic on MacBooks)
3. Start capturing and streaming audio
4. Display transcribed text as it becomes available
"""

import asyncio
import websockets
import numpy as np
import samplerate
import pyaudio
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import sys
import os
import platform
from collections import deque

from volume_window import VolumeWindow
from audio_core import AudioCore
from llm_client import LLMClient
from audio_output import AudioOutput

import json

# Load configuration
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

# Server configuration
SERVER_URI = "ws://10.5.2.10:8765"  # Or ws://<server-ip>:8765 if remote

# Trigger word configuration
TRIGGER_WORD = CONFIG['assistant']['name']

################################################################################
# ASYNCHRONOUS TASKS
################################################################################

async def record_and_send_audio(websocket, volume_window):
    """
    Continuously read audio from the microphone and send raw PCM frames to the server.
    The audio is automatically resampled to 16kHz if the device doesn't support it directly.
    Updates the volume window with current audio levels.
    """
    p = None
    stream = None
    retry_count = 0
    max_retries = 3
    while retry_count < max_retries:
        last_print_time = 0  # For throttling debug output
        try:
            audio_core = AudioCore()
            p, device_info, rate, needs_resampling = audio_core.init_audio_device()
            
            try:
                # Configure stream with optimal Linux ALSA settings
                buffer_size = audio_core.CHUNK * 4  # Larger buffer for stability
                stream = p.open(
                    format=audio_core.FORMAT,
                    channels=1,
                    rate=rate,
                    input=True,
                    input_device_index=device_info['index'],
                    frames_per_buffer=buffer_size,
                    start=False,  # Don't start yet
                    stream_callback=None,  # Use blocking mode for better stability
                    input_host_api_specific_stream_info=None  # Let ALSA handle defaults
                )
                
                # Configure stream parameters
                stream.start_stream()
            except OSError as e:
                print(f"Error opening stream with default settings: {e}")
                print("Trying alternative configuration...")
                
                # Try alternative configuration with default ALSA buffer size
                stream = p.open(
                    format=audio_core.FORMAT,
                    channels=1,
                    rate=rate,
                    input=True,
                    input_device_index=device_info['index'],
                    frames_per_buffer=audio_core.CHUNK,  # Use default chunk size
                    start=True,
                    stream_callback=None  # Use blocking mode
                )

            print(f"\nSuccessfully initialized audio device: {device_info['name']}")
            print(f"Recording at: {rate} Hz")
            print(f"Channels: 1")
            print("\nStart speaking...")

            # Reset retry count on successful initialization
            retry_count = 0
            
            # Only create resampler if needed
            resampler = None
            if needs_resampling:
                resampler = samplerate.Resampler('sinc_best')
                ratio = 16000 / rate

            while True:
                try:
                    # Read with larger timeout and handle overflows
                    try:
                        data = stream.read(audio_core.CHUNK, exception_on_overflow=False)
                    except OSError as e:
                        print(f"Stream read error (trying to recover): {e}")
                        await asyncio.sleep(0.1)  # Give the stream time to recover
                        continue
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    
                    # Convert to float for volume window processing
                    float_data = audio_data.astype(np.float32) / 32768.0
                    
                    # Update volume window with audio data
                    volume_window.process_audio(float_data)
                    
                    # Small delay to allow GUI to update
                    await asyncio.sleep(0.001)
                    
                    # Resample if needed and send to server
                    if needs_resampling:
                        try:
                            # Resample to 16kHz
                            ratio = 16000 / rate
                            resampled_data = resampler.process(
                                float_data,
                                ratio,
                                end_of_input=False
                            )
                            final_data = np.clip(resampled_data * 32768.0, -32768, 32767).astype(np.int16)
                        except Exception as e:
                            print(f"Error during resampling: {e}")
                            continue
                    else:
                        final_data = np.clip(float_data * 32768.0, -32768, 32767).astype(np.int16)
                    
                    # Send the audio
                    await websocket.send(final_data.tobytes())
                except OSError as e:
                    print(f"Error reading from stream: {e}")
                    await asyncio.sleep(0.1)  # Brief pause before retrying
                    continue
                except Exception as e:
                    print(f"Unexpected error during recording: {e}")
                    break

        except asyncio.CancelledError:
            # Task was cancelled (we're shutting down)
            break
        except Exception as e:
            retry_count += 1
            print(f"\nError initializing audio (attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count < max_retries:
                print("Retrying in 2 seconds...")
                await asyncio.sleep(2)
            else:
                print("\nFailed to initialize audio after multiple attempts.")
                print("Please check your audio devices and permissions.")
                break
        finally:
            # Cleanup
            if stream is not None:
                stream.stop_stream()
                stream.close()
            if p is not None:
                p.terminate()



async def receive_transcripts(websocket):
    """
    Continuously receive transcripts from the server and print them in the client console.
    Also detects if the trigger word appears in the first 3 words of the transcript.
    When triggered, sends the transcript to the LLM for processing and handles TTS playback.
    """
    try:
        llm_client = LLMClient()
        audio_output = AudioOutput()
        # Initialize audio output immediately
        audio_output.initialize()
        
        # Define callback for LLM to send TTS requests
        async def handle_llm_chunk(text):
            await websocket.send(f"TTS:{text}")
        
        while True:
            msg = await websocket.recv()
            
            if isinstance(msg, bytes):
                if msg.startswith(b'TTS:'):
                    # Queue audio chunk immediately for playback
                    audio_output.play_chunk(msg)
                elif msg == b'TTS_END':
                    pass
                continue
            
            # Handle text messages
            print(f"\nTranscript: {msg}")
            
            if msg == "TTS_ERROR":
                print("\n[ERROR] TTS generation failed")
                continue
            
            # Check if trigger word appears anywhere in the message
            msg_lower = msg.lower()
            trigger_pos = msg_lower.find(TRIGGER_WORD.lower())
            
            if trigger_pos != -1:
                # Extract everything from the trigger word to the end
                trigger_text = msg[trigger_pos:]
                print(f"\n[TRIGGER DETECTED] Found trigger word: {trigger_text}")
                
                # Process with LLM and stream responses to TTS
                print("\n[AI RESPONSE] ", end="", flush=True)
                
                # Start audio stream before processing to ensure it's ready
                audio_output.start_stream()
                
                # Create non-blocking task for LLM processing
                asyncio.create_task(llm_client.process_trigger(trigger_text, callback=handle_llm_chunk))
    except websockets.ConnectionClosed:
        print("Server closed connection.")

def create_qt_app():
    """Create and configure Qt application"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setStyle('Fusion')
    return app

async def update_gui(volume_window, app):
    """Update GUI periodically"""
    try:
        while volume_window and volume_window.running:
            if volume_window.has_gui:
                # Process Qt events in the main thread
                app.processEvents()
                # Give other tasks a chance to run
                await asyncio.sleep(0)
    except Exception as e:
        print(f"GUI update error: {e}")

async def main(app):
    """Main coroutine that handles audio streaming and transcription"""
    # Set up macOS specific configurations
    if platform.system() == 'Darwin':
        os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

    volume_window = None
    retry_count = 0
    max_retries = 3

    try:
        # Initialize audio
        audio_core = AudioCore()
        p, device_info, rate, needs_resampling = audio_core.init_audio_device()
        
        # Create volume window
        volume_window = VolumeWindow(device_info['name'])
        
        while retry_count < max_retries:
            try:
                print(f"\nAttempting to connect to server at {SERVER_URI} (attempt {retry_count + 1}/{max_retries})...")
                
                # Connect to the server
                async with websockets.connect(SERVER_URI) as websocket:
                    print(f"Connected to server at {SERVER_URI}.")

                    # Create base tasks
                    tasks = [
                        asyncio.create_task(record_and_send_audio(websocket, volume_window)),
                        asyncio.create_task(receive_transcripts(websocket))
                    ]
                    
                    # Add GUI update task
                    if volume_window and volume_window.has_gui:
                        tasks.append(asyncio.create_task(update_gui(volume_window, app)))

                    # Run all tasks until any one completes/fails
                    done, pending = await asyncio.wait(
                        tasks,
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    # Cancel remaining tasks
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                            
                    # If we get here, connection was successful
                    break
                    
            except (ConnectionRefusedError, OSError) as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Connection failed: {str(e)}")
                    print(f"Retrying in 2 seconds...")
                    await asyncio.sleep(2)
                else:
                    print(f"\nFailed to connect after {max_retries} attempts.")
                    print("Please check if the server is running and the SERVER_URI is correct.")
                    break
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Ensure volume window is closed if it exists
        if volume_window:
            volume_window.close()

def run_client():
    """Run the client with proper event loop handling"""
    try:
        # Create Qt application first
        app = create_qt_app()
        
        # Create new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Create and run the main coroutine
        future = asyncio.ensure_future(main(app))
        
        # Run both event loops
        while not future.done():
            # Process Qt events
            app.processEvents()
            # Run one iteration of the asyncio event loop
            loop.stop()
            loop.run_forever()
        
        # Clean up
        loop.run_until_complete(future)
        loop.close()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # Ensure Qt application is properly closed
        if QApplication.instance():
            QApplication.instance().quit()

if __name__ == "__main__":
    run_client()
