

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
    
    
    def __init__(self, model: str, **kwargs):
        
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
        
        self.api_base = kwargs.get("api_base") or os.environ.get(
            "GROK_BASE_URL", "https://api.x.ai/v1"
        )
        
        self.max_tokens = kwargs.get("max_tokens", 1024)
        self.temperature = kwargs.get("temperature", 0.7)
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(self, prompt: str, **kwargs) -> str:
        
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
        
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results
    
    def get_available_models(self) -> List[str]:
        
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            raise ProviderError(f"Failed to get available models: {str(e)}")
    
    def to_dspy(self) -> dspy.LM:
        
        try:
            from dspy.openai import OpenAI as DSPYOpenAI
            
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
