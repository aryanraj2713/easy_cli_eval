"""
Base evaluator interface for LLM Benchmark CLI.

This module contains the abstract base class for evaluators.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..providers.base import BaseLLMProvider


class BaseEvaluator(ABC):
    """Base class for evaluation methods."""
    
    def __init__(self, provider: BaseLLMProvider, **kwargs):
        """
        Initialize the evaluator.
        
        Args:
            provider: The LLM provider to use
            **kwargs: Additional evaluator-specific parameters
        """
        self.provider = provider
        self.kwargs = kwargs
    
    @abstractmethod
    def evaluate(self, model: BaseLLMProvider, dataset: str, config: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a model on a dataset.
        
        Args:
            model: The LLM provider to evaluate
            dataset: The dataset to evaluate on
            config: Evaluation configuration
            
        Returns:
            Dict mapping metric names to scores
            
        Raises:
            EvaluationError: If evaluation fails
        """
        pass
    
    @abstractmethod
    def get_required_config(self) -> List[str]:
        """
        Return list of required configuration parameters.
        
        Returns:
            List of parameter names required for evaluation
        """
        pass
