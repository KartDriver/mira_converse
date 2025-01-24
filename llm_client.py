#!/usr/bin/env python3
"""
LLM Client for handling interactions with vLLM server using OpenAI-compatible API.
Maintains conversation context with automatic timeout and manual reset capabilities.
"""

import json
import os
import asyncio
import time
from openai import AsyncOpenAI
from typing import Optional, List, Dict
from token_counter import estimate_messages_tokens

class LLMClient:
    """Client for interacting with vLLM server using OpenAI-compatible API."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize LLM client with configuration."""
        self.config = self._load_config(config_path)
        # Use API key from config, defaulting to "not-needed" if not present
        api_key = self.config["llm"].get("api_key", "not-needed")
        self.client = AsyncOpenAI(
            base_url=self.config["llm"]["api_base"],
            api_key=api_key
        )
        # Initialize conversation history
        self.conversation_history: List[Dict[str, str]] = []
        self.last_message_time: float = 0
        self.context_timeout: float = 180  # 3 minutes in seconds
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _check_context_timeout(self) -> bool:
        """Check if the conversation context has timed out (more than 3 minutes since last message)."""
        if not self.last_message_time:
            return False
        return (time.time() - self.last_message_time) > self.context_timeout

    def reset_context(self):
        """Manually reset the conversation context."""
        self.conversation_history = []
        self.last_message_time = 0
        print("\n[Context Reset] Conversation history cleared.")

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
            # Check for manual reset trigger
            if "reset context" in transcript.lower():
                self.reset_context()
                if callback:
                    await callback("Context has been reset. How can I help you?")
                return

            # Check for timeout and reset if needed
            if self._check_context_timeout():
                self.reset_context()
                print("\n[Context Timeout] Conversation history cleared due to inactivity.")

            # Update last message time
            self.last_message_time = time.time()

            # Create system prompt with assistant name from config
            system_prompt = f"You are {self.config['assistant']['name']}, a helpful AI assistant who communicates through voice. Important instructions for your responses: 1) Provide only plain text that will be converted to speech - never use markdown, asterisk *, code blocks, or special formatting. 2) Use natural, conversational language as if you're speaking to someone. 3) Never use bullet points, numbered lists, or special characters. 4) Keep responses concise and clear since they will be spoken aloud. 5) Express lists or multiple points in a natural spoken way using words like 'first', 'also', 'finally', etc. 6) Use punctuation only for natural speech pauses (periods, commas, question marks)."

            # Prepare messages with conversation history
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": transcript})

            # Log full context for debugging
            print("\n[Context Debug] Full conversation context being sent to server:")
            for idx, msg in enumerate(messages):
                print(f"\n{idx}. Role: {msg['role']}")
                print(f"   Content: {msg['content']}")
            print("\n[End Context Debug]")

            # Create chat completion request with streaming
            stream = await self.client.chat.completions.create(
                model=self.config["llm"]["model_path"],
                messages=messages,
                temperature=0.7,
                max_tokens=1024,  # Increased from 150 to allow longer responses
                stream=True
            )
            
            # Store the user's message in history before making the request
            self.conversation_history.append({"role": "user", "content": transcript})
            
            # Stream the response while collecting it
            buffer = ""
            full_response = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    buffer += content
                    full_response += content
                    
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
                
            # Store the assistant's response in history
            self.conversation_history.append({"role": "assistant", "content": full_response})

            # Check token count and trim history if needed
            MAX_TOKENS = 8000
            while True:
                # Get all messages including system prompt
                all_messages = [{"role": "system", "content": system_prompt}]
                all_messages.extend(self.conversation_history)
                
                # Estimate total tokens
                total_tokens = estimate_messages_tokens(all_messages)
                
                if total_tokens <= MAX_TOKENS:
                    break
                    
                # Remove oldest exchange (user + assistant messages) if we're over the limit
                if len(self.conversation_history) >= 2:
                    print("\n[Context Trim] Removing oldest messages to stay within token limit")
                    self.conversation_history = self.conversation_history[2:]
                else:
                    # If we somehow still exceed tokens with just the latest exchange,
                    # we have to clear everything
                    print("\n[Context Reset] Message too large, clearing history")
                    self.conversation_history = []
                    break

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
