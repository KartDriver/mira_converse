#!/usr/bin/env python3

import asyncio
import websockets
import torch
import io
import numpy as np
import soundfile as sf
from scipy import signal
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time
from collections import deque

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
# AUDIO PROCESSING
################################################################################

class AudioProcessor:
    def __init__(self):
        # Professional audio time constants
        self.peak_attack = 0.001   # 1ms peak attack
        self.peak_release = 0.100  # 100ms peak release
        self.rms_attack = 0.030    # 30ms RMS attack
        self.rms_release = 0.500   # 500ms RMS release
        
        # Pre-roll buffer for catching speech starts
        self.preroll_duration = 0.3  # seconds
        self.preroll_samples = int(16000 * self.preroll_duration)  # at 16kHz
        self.preroll_buffer = deque(maxlen=self.preroll_samples)
        self.last_speech_end = 0  # timestamp of last speech end
        
        # Level detection
        self.peak_level = -96.0
        self.rms_level = -96.0
        self.last_update = time.time()
        
        # Advanced noise floor tracking from audio_calibration.py
        self.noise_floor = -50.0
        self.min_floor = -65.0
        self.max_floor = -20.0
        self.window_size = 150    # 3 seconds at 50Hz updates
        self.recent_levels = deque(maxlen=self.window_size)
        self.min_history = deque(maxlen=20)  # Longer history for stability
        
        # Speech detection with improved hysteresis
        self.speech_threshold_open = 3.0    # Open threshold above floor (dB)
        self.speech_threshold_close = 2.0   # Close threshold above floor (dB)
        self.is_speaking = False
        self.hold_counter = 0
        self.hold_samples = 10     # Hold samples at 50Hz update rate
        self.pre_emphasis = 0.97
        self.prev_sample = 0.0
        
        # Transcript filtering
        self.transcript_history = deque(maxlen=5)
        self.min_speech_duration = 0.3  # seconds
        
    def bytes_to_float32_audio(self, raw_bytes):
        """Convert raw bytes to float32 audio data"""
        audio_buffer = io.BytesIO(raw_bytes)
        audio_data, sample_rate = sf.read(audio_buffer, dtype='float32', 
                                        format='RAW', subtype='PCM_16', 
                                        samplerate=16000, channels=1)
        return audio_data, sample_rate
    
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

    def process_audio(self, audio_data):
        """Process audio with professional techniques"""
        # Remove DC offset
        dc_removed = audio_data - np.mean(audio_data)
        
        # Apply pre-emphasis filter
        emphasized = np.zeros_like(dc_removed)
        emphasized[0] = dc_removed[0] - self.pre_emphasis * self.prev_sample
        emphasized[1:] = dc_removed[1:] - self.pre_emphasis * dc_removed[:-1]
        self.prev_sample = dc_removed[-1]
        
        # Update levels with envelope following
        now = time.time()
        dt = now - self.last_update
        self.last_update = now
        
        rms = np.sqrt(np.mean(emphasized**2))
        peak = np.max(np.abs(emphasized))
        
        # Convert to dB
        db_rms = 20 * np.log10(max(rms, 1e-10))
        db_peak = 20 * np.log10(max(peak, 1e-10))
        
        # Professional envelope following
        if db_rms > self.rms_level:
            alpha = 1.0 - np.exp(-dt / self.rms_attack)
        else:
            alpha = 1.0 - np.exp(-dt / self.rms_release)
        self.rms_level = self.rms_level + (db_rms - self.rms_level) * alpha
        
        if db_peak > self.peak_level:
            alpha = 1.0 - np.exp(-dt / self.peak_attack)
        else:
            alpha = 1.0 - np.exp(-dt / self.peak_release)
        self.peak_level = self.peak_level + (db_peak - self.peak_level) * alpha
        
        # Spectral analysis for speech detection
        freqs, times, Sxx = signal.spectrogram(emphasized, fs=16000, 
                                             nperseg=256, noverlap=128)
        speech_mask = (freqs >= 100) & (freqs <= 3500)
        speech_energy = np.mean(Sxx[speech_mask, :])
        total_energy = np.mean(Sxx)
        speech_ratio = speech_energy / total_energy if total_energy > 0 else 0
        
        # Calculate zero-crossings
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(emphasized)))) / len(emphasized)
        
        # Debug levels periodically
        current_time = time.time()
        if not hasattr(self, 'last_debug_time'):
            self.last_debug_time = 0
        if current_time - self.last_debug_time >= 1.0:  # Every second
            print(f"\nLevels: RMS={self.rms_level:.1f}dB Floor={self.noise_floor:.1f}dB "
                  f"Ratio={speech_ratio:.2f} ZC={zero_crossings:.3f}")
            self.last_debug_time = current_time
            
        # Quick speech check for floor tracking (basic check)
        quick_speech_check = speech_ratio > 1.05  # Only use ratio for floor protection
        
        # Initialize floor tracking if needed
        if not hasattr(self, 'floor_slow'):
            self.floor_slow = self.rms_level
            
        # Advanced noise floor tracking from audio_calibration.py
        self.recent_levels.append(self.rms_level)
        
        # Get current minimum level estimate (15th percentile for better sensitivity)
        if len(self.recent_levels) > 0:
            current_min = np.percentile(self.recent_levels, 15)
            self.min_history.append(current_min)
            
            # Use weighted average of recent minimums for base floor
            weights = np.exp(-np.arange(len(self.min_history)) / 10)
            base_floor = np.average(self.min_history, weights=weights)
            
            # Professional envelope following for noise floor
            if base_floor < self.noise_floor:
                # Fast attack with smoothing
                alpha_attack = 1.0 - np.exp(-dt / 0.100)  # 100ms attack
                self.noise_floor = max(
                    self.min_floor,
                    self.noise_floor + (base_floor - self.noise_floor) * alpha_attack
                )
            else:
                # Slow release with adaptive time constant
                level_diff = base_floor - self.noise_floor
                release_time = np.interp(level_diff, [0, 20], [2.0, 5.0])
                alpha_release = 1.0 - np.exp(-dt / release_time)
                self.noise_floor = min(
                    self.max_floor,
                    self.noise_floor + (base_floor - self.noise_floor) * alpha_release
                )
        
        # Initialize detection history if needed
        if not hasattr(self, 'detection_history'):
            self.detection_history = deque([False] * 5, maxlen=5)  # Last 5 frames
            self.speech_count = 0  # Count of consecutive speech frames
        
        # Improved speech detection with spectral analysis and hysteresis
        has_speech_character = speech_ratio > 1.03 and zero_crossings > 0.0003
        
        if not self.is_speaking:
            # Check if should open gate
            is_speech = (self.rms_level > self.noise_floor + self.speech_threshold_open and 
                        has_speech_character)
            if is_speech:
                self.is_speaking = True
                self.hold_counter = self.hold_samples
        else:
            # Check if should close gate
            if self.rms_level < self.noise_floor + self.speech_threshold_close:
                if self.hold_counter > 0:
                    self.hold_counter -= 1
                    is_speech = True
                else:
                    self.is_speaking = False
                    is_speech = False
            else:
                self.hold_counter = self.hold_samples
                is_speech = True
        
        # Debug output (focused on decision factors)
        if self.rms_level > self.noise_floor + self.speech_threshold_close - 6:  # Show when getting close
            print(f"Gate Check: Level={self.rms_level:.1f}dB "
                  f"Thresh={self.noise_floor + (self.speech_threshold_open if not self.is_speaking else self.speech_threshold_close):.1f}dB "
                  f"Speech={has_speech_character} "
                  f"-> {is_speech}")
        
        # Professional envelope following (dbx-style)
        if is_speech:
            # Fast attack (5ms)
            attack_time = 0.005
            alpha = 1.0 - np.exp(-dt / attack_time)
            self.speech_count = min(1.0, self.speech_count + alpha)
        else:
            # Slow release (100ms)
            release_time = 0.100
            alpha = 1.0 - np.exp(-dt / release_time)
            self.speech_count = max(0.0, self.speech_count - alpha)
            
        # Simple state tracking
        self.is_speaking = self.speech_count > 0.6  # Slight bias towards closed
        
        # Update timing
        if is_speech or self.is_speaking:
            self.last_speech_time = now
        else:
            self.last_speech_end = now
            
        # Add current audio to pre-roll buffer
        self.add_to_preroll(emphasized)
        
        return {
            'audio': emphasized,
            'is_speech': self.is_speaking,
            'db_level': self.rms_level,
            'noise_floor': self.noise_floor,
            'speech_ratio': speech_ratio,
            'zero_crossings': zero_crossings
        }
    
    def calculate_volume(self, audio_data):
        """Calculate volume using professional audio metering"""
        if self.rms_level > self.noise_floor:
            # Professional audio compression curve
            db_above_floor = self.rms_level - self.noise_floor
            ratio = 0.8  # Subtle compression ratio
            knee = 6.0   # Soft knee width in dB
            
            # Soft knee compression
            if db_above_floor < -knee/2:
                gain = db_above_floor
            elif db_above_floor > knee/2:
                gain = -knee/2 + (db_above_floor - (-knee/2)) / ratio
            else:
                # Smooth transition in knee region
                gain = db_above_floor + ((1/ratio - 1) * 
                       (db_above_floor + knee/2)**2 / (2*knee))
            
            # Convert to linear scale with proper normalization
            volume = np.power(10, gain/20) / np.power(10, (self.max_floor - self.noise_floor)/20)
            return max(0.05, min(1.0, volume))
        return 0.0

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
    processor = AudioProcessor()
    
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
                        audio_data, sr = processor.bytes_to_float32_audio(audio_buffer)
                        result = processor.process_audio(audio_data)
                        
                        # Process speech if detected
                        if result['is_speech']:
                            # Get pre-roll audio if appropriate
                            preroll_audio = processor.get_preroll_audio()
                            
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
                                generate_kwargs={"language": "english"}, 
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
