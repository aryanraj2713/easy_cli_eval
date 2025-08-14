"""
Logging utilities for LLM Benchmark CLI.

This module contains utilities for configuring structured logging.
"""

import os
import sys
from typing import Any, Dict, Optional

import structlog


def configure_logging(level: Optional[str] = None) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
              If None, uses LOG_LEVEL environment variable or defaults to INFO
    """
    # Get log level from environment if not provided
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    
    # Convert string level to int
    level_upper = level.upper()
    numeric_level = getattr(sys.modules["logging"], level_upper, None)
    
    # Configure structlog
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
    
    # Configure stdlib logging
    import logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=numeric_level,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


class LogContext:
    """Context manager for adding context to logs."""
    
    def __init__(self, **kwargs):
        """
        Initialize the log context.
        
        Args:
            **kwargs: Context key-value pairs
        """
        self.kwargs = kwargs
        self.token = None
    
    def __enter__(self):
        """Enter the context and bind the context values."""
        self.token = structlog.contextvars.bind_contextvars(**self.kwargs)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and reset the context values."""
        structlog.contextvars.unbind_contextvars(self.token)
        return False  # Don't suppress exceptions
