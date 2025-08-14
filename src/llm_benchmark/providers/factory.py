

from typing import Dict, Optional, Type

from ..core.exceptions import ConfigurationError
from .base import BaseLLMProvider

class ProviderFactory:
    
    
    _providers: Dict[str, Type[BaseLLMProvider]] = {}
    
    @classmethod
    def register(cls, name: str):
        
        def decorator(provider_cls: Type[BaseLLMProvider]):
            cls._providers[name] = provider_cls
            return provider_cls
        return decorator
    
    @classmethod
    def create(cls, provider: str, model: str, **kwargs) -> BaseLLMProvider:
        
        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ConfigurationError(
                f"Provider '{provider}' not found. Available providers: {available}"
            )
        
        provider_cls = cls._providers[provider]
        return provider_cls(model=model, **kwargs)
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, Type[BaseLLMProvider]]:
        
        return cls._providers.copy()
