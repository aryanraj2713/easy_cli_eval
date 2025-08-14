

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
    
    
    def __init__(self, model: str, **kwargs):
        
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
        
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 1.0)
        self.top_k = kwargs.get("top_k", 40)
        self.max_output_tokens = kwargs.get("max_output_tokens", 1024)
        
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
        
        try:
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
        
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results
    
    def get_available_models(self) -> List[str]:
        
        try:
            models = genai.list_models()
            return [model.name for model in models]
        except Exception as e:
            raise ProviderError(f"Failed to get available models: {str(e)}")
    
    def to_dspy(self) -> dspy.LM:
        
        from ..utils.dspy_wrappers import GeminiDSPyWrapper
        return GeminiDSPyWrapper(
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
    
    def gepa(
        self,
        base_prompt: str,
        target_task: str,
        population_size: int = 10,
        generations: int = 5,
        mutation_rate: float = 0.3,
        **kwargs
    ) -> Dict[str, Any]:
        
        from ..evaluation.gepa import GEPAOptimizer
        
        optimizer = GEPAOptimizer(
            provider=self,
            base_prompt=base_prompt,
            target_task=target_task,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            **kwargs
        )
        
        return optimizer.run()
