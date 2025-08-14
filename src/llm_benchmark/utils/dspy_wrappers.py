

from typing import Any, Dict, List, Optional, Union

import dspy

class GeminiDSPyWrapper(dspy.LM):
    
    
    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        
        super().__init__()
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.genai = genai
            self.client = genai.GenerativeModel(
                model_name=model,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
        except ImportError:
            raise ImportError(
                "Google Generative AI package is required. "
                "Install with: uv pip install -e .[gemini]"
            )
    
    def _call(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens
        
        if generation_config:
            client = self.genai.GenerativeModel(
                model_name=self.model,
                generation_config=generation_config,
            )
        else:
            client = self.client
        
        response = client.generate_content(prompt)
        
        return {
            "choices": [
                {
                    "text": response.text,
                }
            ]
        }
    
    def __call__(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        **kwargs
    ) -> Dict[str, Any]:
        
        if isinstance(prompt, list):
            prompt_str = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in prompt
            ])
        else:
            prompt_str = prompt
        
        return self._call(prompt_str, **kwargs)
