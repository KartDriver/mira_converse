#!/usr/bin/env python3
"""
Audio Chat Client

This script captures audio from your microphone, sends it to a server for real-time
transcription, and displays the transcribed text. It automatically selects the most
appropriate input device and handles audio capture and streaming.

Usage:
    python client.py

The script will automatically:
1. Connect to the transcription server
2. Select the best available microphone
3. Start capturing and streaming audio
4. Display transcribed text as it becomes available
"""

import asyncio
import websockets
import numpy as np
import samplerate
import sys
import os
import platform
from collections import deque
import pygame

from volume_window import VolumeWindow
from audio_core import AudioCore
from llm_client import LLMClient
from audio_output import AudioOutput

import json
import time

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

async def update_volume_window(volume_window):
    """Update pygame window periodically"""
    try:
        while volume_window and volume_window.running:
            # Process pygame events and update display
            volume_window.update()
            # Small delay to prevent high CPU usage
            await asyncio.sleep(0.016)  # ~60 FPS
    except Exception as e:
        print(f"Volume window update error: {e}")

async def record_and_send_audio(websocket, volume_window):
    """
    Continuously read audio from the microphone and send raw PCM frames to the server.
    The audio is automatically resampled to 16kHz if the device doesn't support it directly.
    Updates the volume window with current audio levels.
    """
    retry_count = 0
    max_retries = 3
    while retry_count < max_retries:
        last_print_time = 0  # For throttling debug output
        try:
            audio_core = AudioCore()
            stream, device_info, rate, needs_resampling = audio_core.init_audio_device()
            
            # Update volume window with device name
            if volume_window and volume_window.has_gui:
                volume_window.device_name = device_info['name']
            
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
                    # Read audio with error handling
                    try:
                        audio_data = stream.read(audio_core.CHUNK)[0]
                    except Exception as e:
                        print(f"Stream read error (trying to recover): {e}")
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Update volume window with float32 audio data
                    if volume_window and volume_window.has_gui:
                        volume_window.process_audio(audio_data)
                    
                    # Resample if needed and convert to int16 for server
                    if needs_resampling:
                        try:
                            resampled_data = resampler.process(
                                audio_data,
                                ratio,
                                end_of_input=False
                            )
                            final_data = np.clip(resampled_data * 32768.0, -32768, 32767).astype(np.int16)
                        except Exception as e:
                            print(f"Error during resampling: {e}")
                            continue
                    else:
                        final_data = np.clip(audio_data * 32768.0, -32768, 32767).astype(np.int16)
                    
                    # Send audio data
                    await websocket.send(final_data.tobytes())
                    
                except Exception as e:
                    print(f"Error processing audio chunk: {e}")
                    continue
                
                # Small delay to prevent high CPU usage
                await asyncio.sleep(0.001)
                
        except asyncio.CancelledError:
            print("\nAudio recording cancelled")
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
            if stream is not None:
                try:
                    print("\nClosing input audio stream...")
                    stream.stop()
                    stream.close()
                except Exception as e:
                    print(f"Error closing input stream: {e}")
            # Clean up resampler
            if resampler is not None:
                del resampler

# Initialize audio output globally
audio_output = AudioOutput()

async def receive_transcripts(websocket):
    """
    Continuously receive transcripts from the server and print them in the client console.
    Also detects if the trigger word appears in the transcript.
    When triggered, sends the transcript to the LLM for processing and handles TTS playback.
    """
    try:
        llm_client = LLMClient()
        
        # Define callback for LLM to send TTS requests
        async def handle_llm_chunk(text):
            await websocket.send(f"TTS:{text}")
        
        while True:
            msg = await websocket.recv()
            
            if isinstance(msg, bytes):
                if msg.startswith(b'TTS:'):
                    # Queue audio chunk immediately for playback
                    await audio_output.play_chunk(msg)
                elif msg == b'TTS_END':
                    pass
                continue
            
            # Handle text messages
            if msg == "TTS_ERROR":
                print("\n[ERROR] TTS generation failed")
                continue
            
            # Print transcript with clear formatting
            print(f"\n[TRANSCRIPT] {msg}")
            
            # Check if trigger word appears anywhere in the message
            msg_lower = msg.lower()
            trigger_pos = msg_lower.find(TRIGGER_WORD.lower())
            
            if trigger_pos != -1:
                # Extract everything from the trigger word to the end
                trigger_text = msg[trigger_pos:]
                print(f"\n[TRIGGER DETECTED] Found trigger word: {trigger_text}")
                
                try:
                    # Process with LLM and stream responses to TTS
                    print("\n[AI RESPONSE] ", end="", flush=True)
                    
                    # Start audio stream before processing to ensure it's ready
                    await audio_output.start_stream()
                    
                    # Create non-blocking task for LLM processing
                    asyncio.create_task(llm_client.process_trigger(trigger_text, callback=handle_llm_chunk))
                except Exception as e:
                    print(f"\n[ERROR] Failed to process trigger: {e}")
    except websockets.ConnectionClosed:
        print("Server closed connection.")

async def main():
    """Main coroutine that handles audio streaming and transcription"""
    # Set up macOS specific configurations
    if platform.system() == 'Darwin':
        os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

    volume_window = None
    retry_count = 0
    max_retries = 3

    try:
        # Initialize audio output first
        await audio_output.initialize()
        
        # Create volume window
        volume_window = VolumeWindow("Initializing...")
        
        while retry_count < max_retries:
            try:
                print(f"\nAttempting to connect to server at {SERVER_URI} (attempt {retry_count + 1}/{max_retries})...")
                
                # Connect to the server
                async with websockets.connect(SERVER_URI) as websocket:
                    print(f"Connected to server at {SERVER_URI}.")

                    # Create tasks
                    tasks = [
                        asyncio.create_task(record_and_send_audio(websocket, volume_window)),
                        asyncio.create_task(receive_transcripts(websocket))
                    ]
                    
                    # Add volume window update task if GUI is available
                    if volume_window and volume_window.has_gui:
                        tasks.append(asyncio.create_task(update_volume_window(volume_window)))

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
        try:
            # Clean up audio resources
            if audio_output and audio_output.stream:
                audio_output.close()
            
            # Close volume window if it exists
            if volume_window:
                volume_window.close()
                
            # Cancel any remaining tasks
            for task in asyncio.all_tasks():
                if not task.done() and task != asyncio.current_task():
                    task.cancel()
                    try:
                        asyncio.get_event_loop().run_until_complete(task)
                    except asyncio.CancelledError:
                        pass
        except Exception as e:
            print(f"Error during cleanup: {e}")

def run_client():
    """Run the client"""
    loop = None
    try:
        # Create and run event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
        # Clean up audio resources
        if audio_output and audio_output.stream:
            audio_output.close()
        # Cancel all running tasks
        if loop:
            for task in asyncio.all_tasks(loop):
                task.cancel()
            # Run loop one last time to execute cancellations
            loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(loop), return_exceptions=True))
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        if loop:
            loop.stop()
            loop.close()

if __name__ == "__main__":
    run_client()
