"""
OpenAI provider implementation for LLM Benchmark CLI.
"""

import os
from typing import Any, Dict, List, Optional

import dspy
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.exceptions import APIError, AuthenticationError, ProviderError, RateLimitError
from .base import BaseLLMProvider
from .factory import ProviderFactory

try:
    import openai
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@ProviderFactory.register("openai")
class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, model: str, **kwargs):
        """
        Initialize the OpenAI provider.
        
        Args:
            model: The model name to use (e.g., 'gpt-5', 'gpt-4o', 'gpt-4')
            **kwargs: Additional provider-specific parameters
                - api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
                - api_base: OpenAI API base URL (defaults to OpenAI's default)
                - organization: OpenAI organization ID
                - max_tokens: Maximum tokens to generate
                - temperature: Sampling temperature
        """
        super().__init__(model, **kwargs)
        
        if not HAS_OPENAI:
            raise ImportError(
                "OpenAI package is required. Install with: uv pip install -e .[openai]"
            )
        
        self.api_key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.api_base = kwargs.get("api_base")
        self.organization = kwargs.get("organization")
        
        # Default generation parameters
        self.max_tokens = kwargs.get("max_tokens", 1024)
        self.temperature = kwargs.get("temperature", 0.7)
        
        # Initialize client
        client_kwargs = {"api_key": self.api_key}
        if self.api_base:
            client_kwargs["base_url"] = self.api_base
        if self.organization:
            client_kwargs["organization"] = self.organization
            
        self.client = OpenAI(**client_kwargs)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response from the model.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional generation parameters
                - max_tokens: Maximum tokens to generate
                - temperature: Sampling temperature
                - top_p: Nucleus sampling parameter
                - frequency_penalty: Frequency penalty
                - presence_penalty: Presence penalty
                
        Returns:
            The generated text response
            
        Raises:
            ProviderError: If the API call fails
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", 1.0),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                presence_penalty=kwargs.get("presence_penalty", 0.0),
            )
            return response.choices[0].message.content
        except openai.RateLimitError as e:
            raise RateLimitError(f"OpenAI rate limit exceeded: {str(e)}")
        except openai.AuthenticationError as e:
            raise AuthenticationError(f"OpenAI authentication error: {str(e)}")
        except openai.APIError as e:
            raise APIError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise ProviderError(f"OpenAI provider error: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
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
        # OpenAI doesn't have a native batch API, so we process sequentially
        # In a production system, we'd use async/await for better performance
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results
    
    def get_available_models(self) -> List[str]:
        """
        Return list of available models.
        
        Returns:
            List of model names available from OpenAI
            
        Raises:
            ProviderError: If the API call fails
        """
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            raise ProviderError(f"Failed to get available models: {str(e)}")
    
    def to_dspy(self) -> dspy.LM:
        """
        Convert to a DSPy language model.
        
        Returns:
            A DSPy OpenAI language model instance
        """
        try:
            from dspy.openai import OpenAI as DSPYOpenAI
            
            # Configure with the same parameters
            dspy_kwargs = {
                "model": self.model,
                "api_key": self.api_key,
            }
            if self.api_base:
                dspy_kwargs["api_base"] = self.api_base
                
            return DSPYOpenAI(**dspy_kwargs)
        except ImportError:
            raise ImportError("DSPy is required for this functionality.")
    
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
        Implement Genetic-Evolutionary Prompt Architecture for OpenAI models.
        
        Args:
            base_prompt: Initial prompt template
            target_task: Task description for optimization
            population_size: Number of prompt variants per generation
            generations: Number of evolutionary iterations
            mutation_rate: Probability of prompt mutation
            **kwargs: Additional GAPE parameters
                - fitness_function: Function to evaluate prompt fitness
                - crossover_method: Method for prompt crossover
                - mutation_method: Method for prompt mutation
                
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
        # This is a simplified implementation of GAPE
        # In a production system, this would be more sophisticated
        from ..evaluation.gape import GAPEOptimizer
        
        optimizer = GAPEOptimizer(
            provider=self,
            base_prompt=base_prompt,
            target_task=target_task,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            **kwargs
        )
        
        return optimizer.run()
