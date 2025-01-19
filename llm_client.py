#!/usr/bin/env python3
"""
LLM Client for handling interactions with vLLM server using OpenAI-compatible API.
"""

import json
import os
import asyncio
from openai import AsyncOpenAI
from typing import Optional, List

class LLMClient:
    """Client for interacting with vLLM server using OpenAI-compatible API."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize LLM client with configuration."""
        self.config = self._load_config(config_path)
        self.client = AsyncOpenAI(
            base_url=self.config["llm"]["api_base"],
            api_key="not-needed"  # vLLM doesn't require API key
        )
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    async def process_trigger(self, transcript: str, callback=None):
        """
        Process a triggered transcript with the LLM.
        
        Args:
            transcript: The transcript text to process
            callback: Optional callback function to handle streaming chunks
            
        Returns:
            None, as responses are handled through the callback
        """
        try:
            # Create chat completion request with streaming
            # Create system prompt with assistant name from config
            system_prompt = f"You are {self.config['assistant']['name']}, a helpful AI assistant who communicates through voice. Important instructions for your responses: 1) Provide only plain text that will be converted to speech - never use markdown, code blocks, or special formatting. 2) Use natural, conversational language as if you're speaking to someone. 3) Never use bullet points, numbered lists, or special characters. 4) Keep responses concise and clear since they will be spoken aloud. 5) Express lists or multiple points in a natural spoken way using words like 'first', 'also', 'finally', etc. 6) Use punctuation only for natural speech pauses (periods, commas, question marks)."
            
            stream = await self.client.chat.completions.create(
                model=self.config["llm"]["model_path"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcript}
                ],
                temperature=0.7,
                max_tokens=1024,  # Increased from 150 to allow longer responses
                stream=True
            )
            
            # Stream the response
            buffer = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    buffer += content
                    
                    # Send complete sentences to TTS as they arrive
                    while '.' in buffer or '!' in buffer or '?' in buffer:
                        # Find the last sentence boundary
                        last_period = max(buffer.rfind('.'), buffer.rfind('!'), buffer.rfind('?'))
                        if last_period == -1:
                            break
                            
                        # Extract the complete sentence(s)
                        sentence = buffer[:last_period + 1].strip()
                        if sentence and callback:
                            # Create non-blocking TTS task
                            asyncio.create_task(callback(sentence))
                            
                        # Keep the remainder in the buffer
                        buffer = buffer[last_period + 1:].strip()
            
            # Send any remaining text without waiting
            if buffer.strip() and callback:
                asyncio.create_task(callback(buffer.strip()))
                
            # Print newline after response
            print()
            
        except Exception as e:
            print(f"Error processing LLM request: {e}")
            return None

# Example usage:
if __name__ == "__main__":
    import asyncio
    
    async def test():
        client = LLMClient()
        config = client.config
        response = await client.process_trigger(f"{config['assistant']['name']}, what's the weather like today?")
        print(f"Full response: {response}")
    
    asyncio.run(test())
