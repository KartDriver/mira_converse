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
                print(f"\nResampling configuration:")
                print(f"Source rate: {rate} Hz")
                print(f"Target rate: 16000 Hz")
                print(f"Resampling ratio: {ratio:.4f}")
                print(f"Input chunk size: {CHUNK} samples")
                print(f"Expected output chunk size: ~{int(CHUNK * ratio)} samples")
                print("Using sinc_best resampling quality\n")

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
                    
                    # Calculate current dB level
                    float_data = audio_data.astype(np.float32) / 32768.0
                    rms = np.sqrt(np.mean(float_data**2))
                    current_db = 20 * np.log10(rms) if rms > 0 else -96
                    
                    # Initialize or update noise floor
                    if not hasattr(record_and_send_audio, 'noise_samples'):
                        record_and_send_audio.noise_samples = []
                        record_and_send_audio.last_calibration = 0
                        print("\nCalibrating noise floor...")
                    
                    # Collect samples for 1 second during initialization or recalibration
                    current_time = asyncio.get_event_loop().time()
                    if len(record_and_send_audio.noise_samples) < 50:  # About 1 second of samples
                        record_and_send_audio.noise_samples.append(current_db)
                        if len(record_and_send_audio.noise_samples) == 50:
                            # Calculate noise floor from mean level
                            mean_noise = np.mean(record_and_send_audio.noise_samples)
                            std_noise = np.std(record_and_send_audio.noise_samples)
                            # Set floor slightly below mean to handle background variation
                            record_and_send_audio.noise_floor = mean_noise - 0.1
                            print(f"Noise floor calibrated to: {record_and_send_audio.noise_floor:.1f} dB")
                            record_and_send_audio.last_calibration = current_time
                        await asyncio.sleep(0.001)
                        continue
                    
                    # Recalibrate every 60 seconds if audio level is stable
                    if current_time - record_and_send_audio.last_calibration > 60:
                        if abs(current_db - record_and_send_audio.noise_floor) < 2:  # Stable audio
                            record_and_send_audio.noise_samples = []
                            record_and_send_audio.last_calibration = current_time
                            print("\nRecalibrating noise floor...")
                            continue
                    
                    # Define thresholds with extremely tight ranges
                    NOISE_FLOOR_DB = record_and_send_audio.noise_floor
                    SPEECH_THRESHOLD_DB = NOISE_FLOOR_DB + 0.5  # Just 0.5dB above floor for speech
                    MAX_SPEECH_DB = NOISE_FLOOR_DB + 5  # 5dB range for better sensitivity
                    
                    # Split audio processing into smaller chunks for volume meter
                    chunk_size = 256
                    for i in range(0, len(audio_data), chunk_size):
                        chunk = audio_data[i:i + chunk_size]
                        float_chunk = chunk.astype(np.float32) / 32768.0
                        rms = np.sqrt(np.mean(float_chunk**2))
                        db = 20 * np.log10(rms) if rms > 0 else -96
                        
                        # Calculate volume with dynamic thresholds and aggressive curve
                        if db > NOISE_FLOOR_DB:
                            # Normalize and apply aggressive exponential curve
                            normalized = (db - NOISE_FLOOR_DB) / (MAX_SPEECH_DB - NOISE_FLOOR_DB)
                            # Power of 0.3 makes it rise very quickly at first
                            volume_pct = min(1.0, max(0.0, pow(normalized, 0.3)))
                        else:
                            volume_pct = 0.0
                            
                        # Print levels occasionally for debugging
                        if i == 0:
                            print(f"dB: {db:.1f}, Floor: {NOISE_FLOOR_DB:.1f}, Volume: {volume_pct*100:.0f}%")
                        
                        # Update volume window with percentage value
                        volume_window.update_volume(volume_pct)
                        
                        # Small delay to allow GUI to update
                        await asyncio.sleep(0.001)
                    
                    # Convert to float for processing
                    float_data = audio_data.astype(np.float32) / 32768.0
                    
                    # Initialize state variables if needed
                    if not hasattr(record_and_send_audio, 'state'):
                        record_and_send_audio.state = {
                            'audio_buffer': np.zeros(int(0.3 * rate), dtype=np.float32),  # 0.3s pre-speech buffer
                            'is_speaking': False,
                            'silence_counter': 0,
                            'last_sent_time': 0,
                            'continuous_buffer': []  # Buffer for continuous speech
                        }
                    
                    # Calculate audio levels
                    rms = np.sqrt(np.mean(float_data**2))
                    db = 20 * np.log10(rms) if rms > 0 else -96
                    
                    # Update pre-speech buffer
                    record_and_send_audio.state['audio_buffer'] = np.roll(
                        record_and_send_audio.state['audio_buffer'], 
                        -len(float_data)
                    )
                    record_and_send_audio.state['audio_buffer'][-len(float_data):] = float_data
                    
                    # Speech detection logic
                    if db > SPEECH_THRESHOLD_DB:
                        # Reset silence counter when speech detected
                        record_and_send_audio.state['silence_counter'] = 0
                        
                        if not record_and_send_audio.state['is_speaking']:
                            # Starting to speak - include pre-speech buffer
                            record_and_send_audio.state['is_speaking'] = True
                            record_and_send_audio.state['continuous_buffer'] = [
                                record_and_send_audio.state['audio_buffer'].copy()
                            ]
                        
                        # Add current audio to continuous buffer
                        record_and_send_audio.state['continuous_buffer'].append(float_data)
                        
                        # Send if we've accumulated enough audio (about 1 second)
                        total_samples = sum(len(b) for b in record_and_send_audio.state['continuous_buffer'])
                        if total_samples >= rate:  # 1 second of audio
                            # Combine all buffered audio
                            combined_audio = np.concatenate(record_and_send_audio.state['continuous_buffer'])
                            
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
                                last_chunk = record_and_send_audio.state['continuous_buffer'][-1]
                                record_and_send_audio.state['continuous_buffer'] = [last_chunk[-excess_samples:]]
                            else:
                                record_and_send_audio.state['continuous_buffer'] = []
                    else:
                        # No speech detected - increment silence counter
                        record_and_send_audio.state['silence_counter'] += 1
                        
                        # If silent for more than ~200ms (adjust based on chunk size)
                        if record_and_send_audio.state['silence_counter'] >= 4:
                            if record_and_send_audio.state['is_speaking']:
                                # Send any remaining audio in the buffer
                                if record_and_send_audio.state['continuous_buffer']:
                                    combined_audio = np.concatenate(record_and_send_audio.state['continuous_buffer'])
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
                            record_and_send_audio.state['is_speaking'] = False
                            record_and_send_audio.state['continuous_buffer'] = []
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
