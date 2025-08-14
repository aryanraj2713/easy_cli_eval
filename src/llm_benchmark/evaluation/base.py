

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..providers.base import BaseLLMProvider

class BaseEvaluator(ABC):
    
    
    def __init__(self, provider: BaseLLMProvider, **kwargs):
        
        self.provider = provider
        self.kwargs = kwargs
    
    @abstractmethod
    def evaluate(self, model: BaseLLMProvider, dataset: str, config: Dict[str, Any]) -> Dict[str, float]:
        
        pass
    
    @abstractmethod
    def get_required_config(self) -> List[str]:
        
        pass
