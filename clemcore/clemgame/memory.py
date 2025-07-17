from collections import deque
from copy import deepcopy
from typing import List, Dict, Optional, Union


class SlidingWindowMemory:
    """
    A memory manager that maintains a sliding window of recent conversation messages
    while always preserving the initial prompt.
    
    This class implements a FIFO (First-In-First-Out) sliding window where the oldest
    messages are automatically removed when the window size is exceeded, but the 
    initial prompt is always preserved regardless of window size.
    """
    
    def __init__(self, window_size: int, preserve_initial_prompt: bool = True):
        """
        Initialize the sliding window memory.
        
        Args:
            window_size: Maximum number of message pairs (user+assistant) to keep in the sliding window.
                        Note: This doesn't include the initial prompt which is always preserved.
            preserve_initial_prompt: Whether to always preserve the initial prompt (default: True).
        """
        if window_size <= 0:
            raise ValueError("Window size must be positive")
            
        self.window_size = window_size
        self.preserve_initial_prompt = preserve_initial_prompt
        self.initial_prompt: Optional[Dict] = None
        # Use deque with maxlen for automatic FIFO behavior
        # maxlen = window_size * 2 because each turn has user + assistant message
        self.sliding_messages = deque(maxlen=window_size * 2)
        
    def set_initial_prompt(self, initial_prompt: Union[str, Dict]) -> None:
        """
        Set the initial prompt that will always be preserved.
        
        Args:
            initial_prompt: The initial prompt as string or dict format.
        """
        if isinstance(initial_prompt, str):
            self.initial_prompt = {"role": "user", "content": initial_prompt}
        elif isinstance(initial_prompt, dict):
            self.initial_prompt = deepcopy(initial_prompt)
        else:
            raise ValueError("Initial prompt must be string or dict")
            
    def add_message_pair(self, context: Dict, response: Dict) -> None:
        """
        Add a context-response message pair to the sliding window.
        
        Args:
            context: The user/context message dict with 'role' and 'content'.
            response: The assistant response message dict with 'role' and 'content'.
        """
        if not isinstance(context, dict) or not isinstance(response, dict):
            raise ValueError("Messages must be dictionaries")
            
        # Add to sliding window (deque automatically removes oldest if maxlen exceeded)
        self.sliding_messages.append(deepcopy(context))
        self.sliding_messages.append(deepcopy(response))
        
    def add_single_message(self, message: Dict) -> None:
        """
        Add a single message to the sliding window.
        
        Args:
            message: Message dict with 'role' and 'content'.
        """
        if not isinstance(message, dict):
            raise ValueError("Message must be a dictionary")
            
        self.sliding_messages.append(deepcopy(message))
        
    def get_prompt_messages(self, current_context: Optional[Dict] = None) -> List[Dict]:
        """
        Get the complete message history for model prompting.
        
        This includes:
        1. Initial prompt (if preserved)
        2. Messages from sliding window
        3. Current context (if provided)
        
        Args:
            current_context: Optional current context to append (not stored in memory).
            
        Returns:
            List of message dicts ready for model consumption.
        """
        prompt_messages = []
        
        # Always include initial prompt if preserved and set
        if self.preserve_initial_prompt and self.initial_prompt is not None:
            prompt_messages.append(deepcopy(self.initial_prompt))
            
        # Add messages from sliding window
        prompt_messages.extend([deepcopy(msg) for msg in self.sliding_messages])
        
        # Add current context if provided
        if current_context is not None:
            prompt_messages.append(deepcopy(current_context))
            
        return prompt_messages
        
    def get_all_messages(self) -> List[Dict]:
        """
        Get all stored messages (initial prompt + sliding window).
        
        Returns:
            List of all stored message dicts.
        """
        return self.get_prompt_messages()
        
    def clear(self) -> None:
        """Clear all messages from sliding window but preserve initial prompt."""
        self.sliding_messages.clear()
        
    def clear_all(self) -> None:
        """Clear everything including initial prompt."""
        self.sliding_messages.clear()
        self.initial_prompt = None
        
    def __len__(self) -> int:
        """Return total number of messages stored (including initial prompt)."""
        count = len(self.sliding_messages)
        if self.preserve_initial_prompt and self.initial_prompt is not None:
            count += 1
        return count
        
    def get_window_usage(self) -> Dict:
        """
        Get information about current window usage.
        
        Returns:
            Dict with usage statistics.
        """
        return {
            "current_size": len(self.sliding_messages),
            "max_size": self.sliding_messages.maxlen,
            "window_full": len(self.sliding_messages) == self.sliding_messages.maxlen,
            "has_initial_prompt": self.initial_prompt is not None,
            "total_messages": len(self)
        }
        
    def to_dict(self) -> Dict:
        """
        Serialize the memory state to a dictionary.
        
        Returns:
            Dict representation of memory state.
        """
        return {
            "window_size": self.window_size,
            "preserve_initial_prompt": self.preserve_initial_prompt,
            "initial_prompt": deepcopy(self.initial_prompt) if self.initial_prompt else None,
            "sliding_messages": list(self.sliding_messages),
            "usage": self.get_window_usage()
        } 