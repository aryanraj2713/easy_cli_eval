

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import dspy

class BaseLLMProvider(ABC):
    
    
    def __init__(self, model: str, **kwargs):
        
        self.model = model
        self.kwargs = kwargs
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        
        pass
    
    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        
        pass
    
    @abstractmethod
    def to_dspy(self) -> dspy.LM:
        
        pass
    
    @abstractmethod
    def gepa(
        self,
        base_prompt: str,
        target_task: str,
        population_size: int = 10,
        generations: int = 5,
        mutation_rate: float = 0.3,
        **kwargs
    ) -> Dict[str, Any]:
        
        pass
