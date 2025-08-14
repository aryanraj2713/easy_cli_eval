"""
Helper utilities for LLM Benchmark CLI.

This module contains miscellaneous helper functions used throughout the application.
"""

import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

from .logging import get_logger

logger = get_logger(__name__)

# Type variable for generic function
T = TypeVar("T")


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to time function execution.
    
    Args:
        func: The function to time
        
    Returns:
        Wrapped function that logs execution time
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(
            "function_timing",
            function=func.__name__,
            duration=end_time - start_time,
            duration_ms=(end_time - start_time) * 1000,
        )
        
        return result
    
    return wrapper


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_api_key(provider: str, key_name: Optional[str] = None) -> Optional[str]:
    """
    Get API key from environment variables.
    
    Args:
        provider: Provider name (e.g., 'openai', 'gemini', 'grok')
        key_name: Custom environment variable name
        
    Returns:
        API key or None if not found
    """
    if key_name:
        return os.environ.get(key_name)
    
    # Default environment variable names
    provider_env_vars = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "grok": "GROK_API_KEY",
    }
    
    env_var = provider_env_vars.get(provider.lower())
    if not env_var:
        return None
    
    return os.environ.get(env_var)


def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        items: List to split
        chunk_size: Maximum size of each chunk
        
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds as a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)}m {int(remaining_seconds)}s"
    else:
        hours = seconds // 3600
        remaining = seconds % 3600
        minutes = remaining // 60
        seconds = remaining % 60
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length, adding ellipsis if truncated.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - 3] + "..."
