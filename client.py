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
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import sys
import os
import platform

from volume_window import VolumeWindow
from audio_utils import init_audio, CHUNK, FORMAT
from audio_calibration import AudioCalibration

# Server configuration
SERVER_URI = "ws://10.5.2.10:8765"  # Or ws://<server-ip>:8765 if remote

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
    calibration = AudioCalibration()
    last_print_time = 0  # For throttling debug output

    while retry_count < max_retries:
        try:
            p, device_info, rate, needs_resampling = init_audio()
            
            try:
                # Configure stream with optimal Linux ALSA settings
                buffer_size = CHUNK * 4  # Larger buffer for stability
                stream = p.open(
                    format=FORMAT,
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
                    format=FORMAT,
                    channels=1,
                    rate=rate,
                    input=True,
                    input_device_index=device_info['index'],
                    frames_per_buffer=CHUNK,  # Use default chunk size
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
                        data = stream.read(CHUNK, exception_on_overflow=False)
                    except OSError as e:
                        print(f"Stream read error (trying to recover): {e}")
                        await asyncio.sleep(0.1)  # Give the stream time to recover
                        continue
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    
                    # Convert to float and process audio
                    float_data = audio_data.astype(np.float32) / 32768.0
                    
                    # Process audio with professional techniques and get speech detection result
                    is_speech, processed_audio = calibration.add_sample(float_data)
                    thresholds = calibration.get_thresholds()
                    
                    # Send updated thresholds to server periodically (every ~1 second)
                    current_time = asyncio.get_event_loop().time()
                    if not hasattr(record_and_send_audio, 'last_threshold_update'):
                        record_and_send_audio.last_threshold_update = 0
                    
                    if current_time - record_and_send_audio.last_threshold_update >= 1.0:
                        # Send open threshold as this is when we start detecting speech
                        await websocket.send(f"SPEECH_THRESHOLD:{thresholds['speech_threshold_open']:.1f}")
                        record_and_send_audio.last_threshold_update = current_time
                        print(f"Floor: {thresholds['noise_floor']:.1f} dB, "
                              f"RMS: {thresholds['rms_level']:.1f} dB, "
                              f"Peak: {thresholds['peak_level']:.1f} dB")
                    
                    # Log speech detection status periodically
                    if not hasattr(record_and_send_audio, 'last_status_time'):
                        record_and_send_audio.last_status_time = 0
                    
                    if current_time - record_and_send_audio.last_status_time >= 2.0:
                        print(f"Speech Detection - Floor: {thresholds['noise_floor']:.1f} dB, "
                              f"Active: {is_speech}, "
                              f"Level: {thresholds['rms_level']:.1f} dB")
                        record_and_send_audio.last_status_time = current_time
                    
                    # Process audio chunks for volume meter with professional metering
                    chunk_size = 256
                    for i in range(0, len(processed_audio), chunk_size):
                        chunk = processed_audio[i:i + chunk_size]
                        
                        # Calculate zero-crossings on processed audio
                        zero_crossings = np.sum(np.abs(np.diff(np.signbit(chunk)))) / len(chunk)
                        
                        # Use professional metering for volume display
                        volume_pct = calibration.calculate_volume(chunk)
                            
                        # Print levels occasionally for debugging (every 2 seconds)
                        if i == 0 and current_time - last_print_time >= 2:
                            print(f"Vol: {volume_pct*100:.0f}%")
                            last_print_time = current_time
                        
                        # Update volume window with percentage value
                        volume_window.update_volume(volume_pct)
                        
                        # Small delay to allow GUI to update
                        await asyncio.sleep(0.001)
                    
                    # Initialize enhanced state variables if needed
                    if not hasattr(record_and_send_audio, 'state'):
                        record_and_send_audio.state = {
                            'audio_buffer': np.zeros(int(0.5 * rate), dtype=np.float32),  # 0.5s pre-speech buffer
                            'is_speaking': False,
                            'silence_counter': 0,
                            'last_sent_time': 0,
                            'continuous_buffer': [],  # Buffer for continuous speech
                            'energy_window': np.zeros(10),  # Rolling window of audio energy
                            'energy_index': 0,
                            'last_energy': 0
                        }
                    
                    state = record_and_send_audio.state
                    
                    # Speech detection and processing with enhanced logic
                    if is_speech or state['is_speaking']:
                        if is_speech:
                            # Reset silence counter when active speech detected
                            state['silence_counter'] = 0
                            
                            if not state['is_speaking']:
                                # Starting to speak - include pre-speech buffer with energy-based trimming
                                state['is_speaking'] = True
                                pre_speech_data = processed_audio  # Use processed audio
                                state['continuous_buffer'] = [pre_speech_data]
                        
                        # Always add current audio while speaking
                        state['continuous_buffer'].append(processed_audio)
                        
                        # Send if we've accumulated enough audio (about 1 second)
                        total_samples = sum(len(b) for b in state['continuous_buffer'])
                        if total_samples >= rate:  # 1 second of audio
                            # Combine all buffered audio
                            combined_audio = np.concatenate(state['continuous_buffer'])
                            
                            if needs_resampling:
                                try:
                                    # Resample the combined audio
                                    ratio = 16000 / rate
                                    resampled_data = resampler.process(
                                        combined_audio,
                                        ratio,
                                        end_of_input=False
                                    )
                                    final_data = np.clip(resampled_data * 32768.0, -32768, 32767).astype(np.int16)
                                except Exception as e:
                                    print(f"Error during resampling: {e}")
                                    continue
                            else:
                                final_data = np.clip(combined_audio * 32768.0, -32768, 32767).astype(np.int16)
                            
                            # Send the audio
                            await websocket.send(final_data.tobytes())
                            
                            # Keep only the most recent audio for the next buffer
                            excess_samples = total_samples - rate
                            if excess_samples > 0:
                                last_chunk = state['continuous_buffer'][-1]
                                state['continuous_buffer'] = [last_chunk[-excess_samples:]]
                            else:
                                state['continuous_buffer'] = []
                    else:
                        # No speech detected - increment silence counter
                        state['silence_counter'] += 1
                        
                        # Dynamic silence detection based on speech duration
                        silence_threshold = 8 if len(state['continuous_buffer']) > 20 else 4
                        if state['silence_counter'] >= silence_threshold:
                            if state['is_speaking']:
                                # Send any remaining audio in the buffer
                                if state['continuous_buffer']:
                                    combined_audio = np.concatenate(state['continuous_buffer'])
                                    if needs_resampling:
                                        try:
                                            ratio = 16000 / rate
                                            resampled_data = resampler.process(
                                                combined_audio,
                                                ratio,
                                                end_of_input=False
                                            )
                                            final_data = np.clip(resampled_data * 32768.0, -32768, 32767).astype(np.int16)
                                            await websocket.send(final_data.tobytes())
                                        except Exception as e:
                                            print(f"Error during final resampling: {e}")
                                    else:
                                        final_data = np.clip(combined_audio * 32768.0, -32768, 32767).astype(np.int16)
                                        await websocket.send(final_data.tobytes())
                            
                            # Reset speaking state
                            state['is_speaking'] = False
                            state['continuous_buffer'] = []
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
    """
    try:
        while True:
            msg = await websocket.recv()  # Wait for text message
            print(f"Transcript: {msg}")
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
        p, device_info, rate, needs_resampling = init_audio()
        
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
