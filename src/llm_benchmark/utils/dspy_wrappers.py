"""
DSPy wrapper utilities for LLM Benchmark CLI.

This module contains wrapper classes for integrating providers with DSPy.
"""

from typing import Any, Dict, List, Optional, Union

import dspy


class GeminiDSPyWrapper(dspy.LM):
    """
    DSPy wrapper for Google Gemini models.
    
    Since DSPy doesn't have native support for Gemini yet, this wrapper
    provides the necessary interface.
    """
    
    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        """
        Initialize the Gemini DSPy wrapper.
        
        Args:
            model: Gemini model name
            api_key: Google API key
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__()
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Import and configure the Gemini client
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
        """
        Call the Gemini model.
        
        Args:
            prompt: The prompt to send to the model
            temperature: Sampling temperature (overrides instance default)
            max_tokens: Maximum tokens to generate (overrides instance default)
            **kwargs: Additional parameters
            
        Returns:
            Dict with model response
        """
        # Update generation config if needed
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens
        
        # Create a new client if generation config changed
        if generation_config:
            client = self.genai.GenerativeModel(
                model_name=self.model,
                generation_config=generation_config,
            )
        else:
            client = self.client
        
        # Generate response
        response = client.generate_content(prompt)
        
        # Format response to match DSPy's expected format
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
        """
        Call the model with a prompt.
        
        Args:
            prompt: The prompt to send to the model (string or chat format)
            **kwargs: Additional parameters
            
        Returns:
            Dict with model response
        """
        # Convert chat format to string if needed
        if isinstance(prompt, list):
            prompt_str = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in prompt
            ])
        else:
            prompt_str = prompt
        
        return self._call(prompt_str, **kwargs)
