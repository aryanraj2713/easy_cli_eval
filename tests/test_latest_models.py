"""
Tests for latest model support.
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from src.llm_benchmark.providers.openai import OpenAIProvider
from src.llm_benchmark.providers.gemini import GeminiProvider
from src.llm_benchmark.providers.grok import GrokProvider


@patch("src.llm_benchmark.providers.openai.OpenAI")
def test_openai_gpt5_support(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "GPT-5 response"
    mock_client.chat.completions.create.return_value = mock_response
    
    provider = OpenAIProvider(model="gpt-5", api_key="test_key")
    
    response = provider.generate("Test prompt")
    
    assert response == "GPT-5 response"
    
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args["model"] == "gpt-5"


@patch("src.llm_benchmark.providers.gemini.genai")
def test_gemini_25_pro_support(mock_genai):
    mock_model = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model
    
    mock_response = MagicMock()
    mock_response.text = "Gemini 2.5 Pro response"
    mock_model.generate_content.return_value = mock_response
    
    mock_genai.configure = MagicMock()
    
    provider = GeminiProvider(model="gemini-2.5-pro", api_key="test_key")
    
    response = provider.generate("Test prompt")
    
    assert response == "Gemini 2.5 Pro response"
    
    mock_genai.GenerativeModel.assert_called_once()
    call_args = mock_genai.GenerativeModel.call_args[1]
    assert call_args["model_name"] == "gemini-2.5-pro"


@patch("src.llm_benchmark.providers.grok.OpenAI")
def test_grok_2_support(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Grok-2 response"
    mock_client.chat.completions.create.return_value = mock_response
    
    provider = GrokProvider(model="grok-2", api_key="test_key")
    
    response = provider.generate("Test prompt")
    
    assert response == "Grok-2 response"
    
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args["model"] == "grok-2"
