"""
Custom exceptions for the LLM Benchmark CLI.

This module contains all custom exceptions used throughout the application.
"""

class LLMBenchmarkError(Exception):
    """Base exception for all LLM Benchmark errors."""
    pass


class ConfigurationError(LLMBenchmarkError):
    """Raised when there is an error in the configuration."""
    pass


class ProviderError(LLMBenchmarkError):
    """Raised when there is an error with an LLM provider."""
    pass


class APIError(ProviderError):
    """Raised when there is an error with an API call."""
    pass


class RateLimitError(APIError):
    """Raised when an API rate limit is exceeded."""
    pass


class AuthenticationError(APIError):
    """Raised when there is an authentication error with an API."""
    pass


class TaskError(LLMBenchmarkError):
    """Raised when there is an error with a benchmark task."""
    pass


class DatasetError(LLMBenchmarkError):
    """Raised when there is an error with a dataset."""
    pass


class EvaluationError(LLMBenchmarkError):
    """Raised when there is an error during evaluation."""
    pass


class MetricError(EvaluationError):
    """Raised when there is an error with a metric calculation."""
    pass
