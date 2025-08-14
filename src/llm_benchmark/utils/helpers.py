

import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

from .logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

def timed(func: Callable[..., T]) -> Callable[..., T]:
    
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
    
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def get_api_key(provider: str, key_name: Optional[str] = None) -> Optional[str]:
    
    if key_name:
        return os.environ.get(key_name)
    
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
    
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def format_duration(seconds: float) -> str:
    
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
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - 3] + "..."
