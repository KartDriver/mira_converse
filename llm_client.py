#!/usr/bin/env python3
"""
LLM Client for handling interactions with vLLM server using OpenAI-compatible API.
"""

import json
import os
from openai import AsyncOpenAI
from typing import Optional

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
    
    async def process_trigger(self, transcript: str) -> Optional[str]:
        """
        Process a triggered transcript with the LLM.
        
        Args:
            transcript: The transcript text to process
            
        Returns:
            The LLM's response text, or None if there was an error
        """
        try:
            # Create chat completion request with streaming
            stream = await self.client.chat.completions.create(
                model=self.config["llm"]["model_path"],
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant named Veronica. Respond naturally to user queries."},
                    {"role": "user", "content": transcript}
                ],
                temperature=0.7,
                max_tokens=150,
                stream=True
            )
            
            # Stream the response
            collected_messages = []
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    collected_messages.append(content)
            
            # Print newline after response
            print()
            
            # Return the complete message
            return "".join(collected_messages)
            
        except Exception as e:
            print(f"Error processing LLM request: {e}")
            return None

# Example usage:
if __name__ == "__main__":
    import asyncio
    
    async def test():
        client = LLMClient()
        response = await client.process_trigger("Veronica, what's the weather like today?")
        print(f"Full response: {response}")
    
    asyncio.run(test())
