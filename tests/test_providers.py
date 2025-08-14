"""
Tests for provider implementations.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.llm_benchmark.core.exceptions import AuthenticationError, ConfigurationError, ProviderError
from src.llm_benchmark.providers.base import BaseLLMProvider
from src.llm_benchmark.providers.factory import ProviderFactory
from src.llm_benchmark.providers.openai import OpenAIProvider


def test_provider_factory_registration():
    class MockProvider(BaseLLMProvider):
        def generate(self, prompt, **kwargs):
            return "Mock response"
        
        def batch_generate(self, prompts, **kwargs):
            return ["Mock response"] * len(prompts)
        
        def get_available_models(self):
            return ["mock-model"]
        
        def to_dspy(self):
            return None
        
        def gepa(self, base_prompt, target_task, **kwargs):
            return {"best_prompt": "Mock prompt"}
    
    ProviderFactory.register("mock")(MockProvider)
    
    providers = ProviderFactory.get_available_providers()
    assert "mock" in providers
    assert providers["mock"] == MockProvider
    
    provider = ProviderFactory.create("mock", "mock-model")
    assert isinstance(provider, MockProvider)
    assert provider.model == "mock-model"


def test_provider_factory_invalid_provider():
    with pytest.raises(ConfigurationError):
        ProviderFactory.create("invalid", "model")


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OpenAI API key not available")
def test_openai_provider_initialization():
    provider = OpenAIProvider(model="gpt-3.5-turbo")
    assert provider.model == "gpt-3.5-turbo"
    assert provider.api_key is not None


@patch("src.llm_benchmark.providers.openai.OpenAI")
def test_openai_provider_generate(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_client.chat.completions.create.return_value = mock_response
    
    provider = OpenAIProvider(model="gpt-3.5-turbo", api_key="test_key")
    
    response = provider.generate("Test prompt")
    
    assert response == "Test response"
    
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args["model"] == "gpt-3.5-turbo"
    assert call_args["messages"] == [{"role": "user", "content": "Test prompt"}]


@patch("src.llm_benchmark.providers.openai.OpenAI")
def test_openai_provider_batch_generate(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_client.chat.completions.create.return_value = mock_response
    
    provider = OpenAIProvider(model="gpt-3.5-turbo", api_key="test_key")
    
    responses = provider.batch_generate(["Prompt 1", "Prompt 2"])
    
    assert responses == ["Test response", "Test response"]
    
    assert mock_client.chat.completions.create.call_count == 2
