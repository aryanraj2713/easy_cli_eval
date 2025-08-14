"""
Base provider interface for LLM Benchmark CLI.

This module contains the abstract base class for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import dspy


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, model: str, **kwargs):
        """
        Initialize the provider.
        
        Args:
            model: The model name to use
            **kwargs: Additional provider-specific parameters
        """
        self.model = model
        self.kwargs = kwargs
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response from the model.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional generation parameters
            
        Returns:
            The generated text response
            
        Raises:
            ProviderError: If the API call fails
        """
        pass
    
    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: The prompts to send to the model
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated text responses
            
        Raises:
            ProviderError: If the API call fails
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Return list of available models.
        
        Returns:
            List of model names available from this provider
            
        Raises:
            ProviderError: If the API call fails
        """
        pass
    
    @abstractmethod
    def to_dspy(self) -> dspy.LM:
        """
        Convert to a DSPy language model.
        
        Returns:
            A DSPy language model instance
            
        Raises:
            ImportError: If DSPy is not installed
        """
        pass
    
    @abstractmethod
    def gape(
        self,
        base_prompt: str,
        target_task: str,
        population_size: int = 10,
        generations: int = 5,
        mutation_rate: float = 0.3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Implement Genetic-Evolutionary Prompt Architecture.
        
        Args:
            base_prompt: Initial prompt template
            target_task: Task description for optimization
            population_size: Number of prompt variants per generation
            generations: Number of evolutionary iterations
            mutation_rate: Probability of prompt mutation
            **kwargs: Additional GAPE parameters
            
        Returns:
            Dict containing:
            - best_prompt: Optimized prompt
            - fitness_scores: Performance metrics across generations
            - evolution_history: Detailed generation-by-generation results
            - final_metrics: Comprehensive evaluation results
            
        Raises:
            ProviderError: If API calls fail
            ConfigurationError: If parameters are invalid
        """
        pass
