

import os
import sys
from typing import Any, Dict, Optional

import structlog

def configure_logging(level: Optional[str] = None) -> None:
    
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    
    level_upper = level.upper()
    numeric_level = getattr(sys.modules["logging"], level_upper, None)
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    import logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=numeric_level,
    )

def get_logger(name: str) -> structlog.BoundLogger:
    
    return structlog.get_logger(name)

class LogContext:
    
    
    def __init__(self, **kwargs):
        
        self.kwargs = kwargs
        self.token = None
    
    def __enter__(self):
        
        self.token = structlog.contextvars.bind_contextvars(**self.kwargs)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        
        structlog.contextvars.unbind_contextvars(self.token)
        return False  
