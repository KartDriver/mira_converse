#!/usr/bin/env python3
"""
Test script for AudioOutput class
"""

import asyncio
import numpy as np
from audio_output import AudioOutput

def generate_test_audio():
    """Generate a simple sine wave as test audio in TTS format"""
    duration = 1.0  # seconds
    sample_rate = 24000  # match TTS rate
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Generate a 440 Hz sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Convert to int16 format (same as TTS output)
    audio = (audio * 32767).astype(np.int16)
    # Add TTS: prefix to match real TTS chunks
    return b'TTS:' + audio.tobytes()

async def main():
    """Test the AudioOutput class"""
    try:
        # Create audio output instance
        audio = AudioOutput()
        
        # Initialize audio
        await audio.initialize()
        
        # Start the playback thread
        await audio.start_stream()
        
        # Generate and queue some test audio
        test_audio = generate_test_audio()
        print("\nPlaying test audio...")
        await audio.play_chunk(test_audio)
        
        # Wait for audio to finish
        await asyncio.sleep(2)
        
        # Clean up
        audio.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
