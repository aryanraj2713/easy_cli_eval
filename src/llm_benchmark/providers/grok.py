"""
Grok/X.AI provider implementation for LLM Benchmark CLI.

This provider uses the OpenAI-compatible API for Grok models.
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


@ProviderFactory.register("grok")
class GrokProvider(BaseLLMProvider):
    """Grok/X.AI provider implementation using OpenAI-compatible API."""
    
    def __init__(self, model: str, **kwargs):
        """
        Initialize the Grok provider.
        
        Args:
            model: The model name to use (e.g., 'grok-2', 'grok-1')
            **kwargs: Additional provider-specific parameters
                - api_key: Grok API key (defaults to GROK_API_KEY env var)
                - api_base: Grok API base URL (defaults to GROK_BASE_URL env var)
                - max_tokens: Maximum tokens to generate
                - temperature: Sampling temperature
        """
        super().__init__(model, **kwargs)
        
        if not HAS_OPENAI:
            raise ImportError(
                "OpenAI package is required for Grok API. Install with: uv pip install -e .[grok]"
            )
        
        self.api_key = kwargs.get("api_key") or os.environ.get("GROK_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                "Grok API key not provided. Set GROK_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Default API base URL for Grok (may change)
        self.api_base = kwargs.get("api_base") or os.environ.get(
            "GROK_BASE_URL", "https://api.x.ai/v1"
        )
        
        # Default generation parameters
        self.max_tokens = kwargs.get("max_tokens", 1024)
        self.temperature = kwargs.get("temperature", 0.7)
        
        # Initialize client
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
    
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
            raise RateLimitError(f"Grok rate limit exceeded: {str(e)}")
        except openai.AuthenticationError as e:
            raise AuthenticationError(f"Grok authentication error: {str(e)}")
        except openai.APIError as e:
            raise APIError(f"Grok API error: {str(e)}")
        except Exception as e:
            raise ProviderError(f"Grok provider error: {str(e)}")
    
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
        # Process sequentially for now
        # In a production system, we'd use async/await for better performance
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results
    
    def get_available_models(self) -> List[str]:
        """
        Return list of available models.
        
        Returns:
            List of model names available from Grok
            
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
            A DSPy OpenAI language model instance configured for Grok
        """
        try:
            from dspy.openai import OpenAI as DSPYOpenAI
            
            # Configure with the same parameters but for Grok's API
            return DSPYOpenAI(
                model=self.model,
                api_key=self.api_key,
                api_base=self.api_base,
            )
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
        Implement Genetic-Evolutionary Prompt Architecture for Grok models.
        
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
            - cost_analysis: Token usage and API costs
            
        Raises:
            ProviderError: If API calls fail
            RateLimitError: If rate limits exceeded
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
