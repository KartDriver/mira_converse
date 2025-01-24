#!/usr/bin/env python3
"""
Token estimation utilities for LLM context management.
Uses simple heuristics to estimate token count for text.
"""

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a piece of text.
    This is a simple estimation based on common tokenization patterns:
    - Most tokenizers treat punctuation as separate tokens
    - Common words are usually single tokens
    - Longer or uncommon words may be split into multiple tokens
    - Spaces are usually their own tokens
    
    Args:
        text: The text to estimate tokens for
        
    Returns:
        Estimated number of tokens
    """
    if not text:
        return 0
        
    # Count words (splitting on whitespace)
    words = text.split()
    token_count = len(words)
    
    # Add tokens for spaces between words
    token_count += len(words) - 1
    
    # Add tokens for punctuation (common in speech: ,.?!;:)
    token_count += sum(1 for char in text if char in ',.?!;:')
    
    # Add extra tokens for longer words that might be split
    # Most tokenizers split words longer than ~12 characters
    token_count += sum(1 for word in words if len(word) > 12)
    
    return token_count

def estimate_messages_tokens(messages: list) -> int:
    """
    Estimate total tokens in a list of chat messages.
    Each message has some overhead for role and formatting.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        
    Returns:
        Estimated total tokens
    """
    total_tokens = 0
    
    for message in messages:
        # Add tokens for message content
        total_tokens += estimate_tokens(message['content'])
        
        # Add overhead tokens for message formatting
        # Each message has role prefix and some JSON formatting
        total_tokens += 4  # Approximate overhead per message
        
    return total_tokens
