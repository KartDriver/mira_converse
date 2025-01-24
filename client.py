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
import tkinter as tk
import threading
import queue

from graphical_interface import AudioInterface
from audio_core import AudioCore
from llm_client import LLMClient
from audio_output import AudioOutput

import json
import time

# Load configuration
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

# Server configuration
SERVER_URI = f"ws://{CONFIG['server']['websocket']['host']}:{CONFIG['server']['websocket']['port']}"

# Trigger word configuration
TRIGGER_WORD = CONFIG['assistant']['name']

################################################################################
# ASYNCHRONOUS TASKS
################################################################################

async def record_and_send_audio(websocket, audio_interface):
    """
    Continuously read audio from the microphone and send raw PCM frames to the server.
    The audio is automatically resampled to 16kHz if the device doesn't support it directly.
    Updates the audio interface with current audio levels.
    """
    retry_count = 0
    max_retries = 3
    audio_core = None
    stream = None
    
    while retry_count < max_retries:
        try:
            audio_core = AudioCore()
            stream, device_info, rate, needs_resampling = audio_core.init_audio_device()
            
            # Update audio interface with device name
            if audio_interface and audio_interface.has_gui:
                audio_interface.input_device_queue.put(device_info['name'])
            
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
                    
                    # Update audio interface with float32 audio data
                    if audio_interface and audio_interface.has_gui:
                        audio_interface.process_audio(audio_data)
                    
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

class AsyncThread(threading.Thread):
    def __init__(self, audio_interface):
        super().__init__()
        self.audio_interface = audio_interface
        self.loop = None
        self.websocket = None
        self.tasks = []
        self.running = True
        self.daemon = True
    
    async def connect_to_server(self):
        """Connect to WebSocket server"""
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                print(f"\nAttempting to connect to server at {SERVER_URI} (attempt {retry_count + 1}/{max_retries})...")
                self.websocket = await websockets.connect(SERVER_URI)
                print(f"Connected to server at {SERVER_URI}.")
                return True
            except (ConnectionRefusedError, OSError) as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Connection failed: {str(e)}")
                    print(f"Retrying in 2 seconds...")
                    await asyncio.sleep(2)
                else:
                    print(f"\nFailed to connect after {max_retries} attempts.")
                    print("Please check if the server is running and the SERVER_URI is correct.")
                    return False
    
    async def async_main(self):
        """Main async loop"""
        try:
            # Initialize audio output
            await audio_output.initialize()
            
            # Connect to server
            if not await self.connect_to_server():
                return
            
            # Create tasks
            self.tasks = [
                asyncio.create_task(record_and_send_audio(self.websocket, self.audio_interface)),
                asyncio.create_task(receive_transcripts(self.websocket))
            ]
            
            # Wait for tasks to complete
            await asyncio.gather(*self.tasks)
            
        except Exception as e:
            print(f"Error in async main: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            # Clean up audio resources
            if audio_output and audio_output.stream:
                audio_output.close()
            
            # Close websocket
            if self.websocket:
                await self.websocket.close()
            
            # Cancel tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def run(self):
        """Run the async event loop in this thread"""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.async_main())
        except Exception as e:
            print(f"Error in async thread: {e}")
        finally:
            try:
                if self.loop and not self.loop.is_closed():
                    self.loop.stop()
                    self.loop.close()
            except Exception as e:
                print(f"Error closing event loop: {e}")

def run_client():
    """Run the client"""
    try:
        # Set up macOS specific configurations
        if platform.system() == 'Darwin':
            os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
            os.environ['TK_SILENCE_DEPRECATION'] = '1'
        
        # Initialize audio output
        audio_output.initialize_sync()
        
        # Create audio interface in main thread
        audio_interface = AudioInterface(
            input_device_name="Initializing...",
            output_device_name=audio_output.get_device_name(),
            on_input_change=None,  # Will be handled by AudioCore
            on_output_change=audio_output.set_device_by_name
        )
        
        # Create and start async thread
        async_thread = AsyncThread(audio_interface)
        async_thread.start()
        
        # Run tkinter mainloop in main thread
        audio_interface.root.mainloop()
        
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    run_client()
