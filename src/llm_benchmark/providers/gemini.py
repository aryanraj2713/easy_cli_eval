"""
Google Gemini provider implementation for LLM Benchmark CLI.
"""

import os
from typing import Any, Dict, List, Optional

import dspy
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.exceptions import APIError, AuthenticationError, ProviderError, RateLimitError
from .base import BaseLLMProvider
from .factory import ProviderFactory

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


@ProviderFactory.register("gemini")
class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider implementation."""
    
    def __init__(self, model: str, **kwargs):
        """
        Initialize the Gemini provider.
        
        Args:
            model: The model name to use (e.g., 'gemini-2.5-pro', 'gemini-1.5-pro', 'gemini-pro')
            **kwargs: Additional provider-specific parameters
                - api_key: Google API key (defaults to GOOGLE_API_KEY env var)
                - temperature: Sampling temperature
                - top_p: Nucleus sampling parameter
                - top_k: Top-k sampling parameter
                - max_output_tokens: Maximum tokens to generate
        """
        super().__init__(model, **kwargs)
        
        if not HAS_GEMINI:
            raise ImportError(
                "Google Generative AI package is required. "
                "Install with: uv pip install -e .[gemini]"
            )
        
        self.api_key = kwargs.get("api_key") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                "Google API key not provided. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Default generation parameters
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 1.0)
        self.top_k = kwargs.get("top_k", 40)
        self.max_output_tokens = kwargs.get("max_output_tokens", 1024)
        
        # Initialize client
        genai.configure(api_key=self.api_key)
        self.generation_config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
        }
    
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
                - temperature: Sampling temperature
                - top_p: Nucleus sampling parameter
                - top_k: Top-k sampling parameter
                - max_output_tokens: Maximum tokens to generate
                
        Returns:
            The generated text response
            
        Raises:
            ProviderError: If the API call fails
        """
        try:
            # Update generation config with any kwargs
            generation_config = self.generation_config.copy()
            if "temperature" in kwargs:
                generation_config["temperature"] = kwargs["temperature"]
            if "top_p" in kwargs:
                generation_config["top_p"] = kwargs["top_p"]
            if "top_k" in kwargs:
                generation_config["top_k"] = kwargs["top_k"]
            if "max_output_tokens" in kwargs:
                generation_config["max_output_tokens"] = kwargs["max_output_tokens"]
            
            model = genai.GenerativeModel(model_name=self.model, 
                                         generation_config=generation_config)
            response = model.generate_content(prompt)
            
            if response.text:
                return response.text
            else:
                raise ProviderError("Gemini returned empty response")
                
        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                raise RateLimitError(f"Gemini rate limit exceeded: {str(e)}")
            elif "auth" in str(e).lower() or "key" in str(e).lower():
                raise AuthenticationError(f"Gemini authentication error: {str(e)}")
            else:
                raise ProviderError(f"Gemini provider error: {str(e)}")
    
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
        # Gemini doesn't have a native batch API, so we process sequentially
        # In a production system, we'd use async/await for better performance
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results
    
    def get_available_models(self) -> List[str]:
        """
        Return list of available models.
        
        Returns:
            List of model names available from Gemini
            
        Raises:
            ProviderError: If the API call fails
        """
        try:
            models = genai.list_models()
            return [model.name for model in models]
        except Exception as e:
            raise ProviderError(f"Failed to get available models: {str(e)}")
    
    def to_dspy(self) -> dspy.LM:
        """
        Convert to a DSPy language model.
        
        Returns:
            A DSPy language model instance
        """
        # DSPy doesn't have a native Gemini integration yet
        # We'll use a custom wrapper
        from ..utils.dspy_wrappers import GeminiDSPyWrapper
        return GeminiDSPyWrapper(
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
    
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
        Implement Genetic-Evolutionary Prompt Architecture for Gemini models.
        
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
