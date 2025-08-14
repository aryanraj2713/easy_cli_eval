"""
Provider factory for LLM Benchmark CLI.

This module contains the factory for creating LLM providers.
"""

from typing import Dict, Optional, Type

from ..core.exceptions import ConfigurationError
from .base import BaseLLMProvider


class ProviderFactory:
    """Factory for creating LLM providers."""
    
    _providers: Dict[str, Type[BaseLLMProvider]] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Register a provider class with the factory.
        
        Args:
            name: The name of the provider
            
        Returns:
            Decorator function
        """
        def decorator(provider_cls: Type[BaseLLMProvider]):
            cls._providers[name] = provider_cls
            return provider_cls
        return decorator
    
    @classmethod
    def create(cls, provider: str, model: str, **kwargs) -> BaseLLMProvider:
        """
        Create a provider instance.
        
        Args:
            provider: The provider name
            model: The model name
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Provider instance
            
        Raises:
            ConfigurationError: If the provider is not registered
        """
        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ConfigurationError(
                f"Provider '{provider}' not found. Available providers: {available}"
            )
        
        provider_cls = cls._providers[provider]
        return provider_cls(model=model, **kwargs)
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, Type[BaseLLMProvider]]:
        """
        Get all registered providers.
        
        Returns:
            Dict mapping provider names to provider classes
        """
        return cls._providers.copy()
