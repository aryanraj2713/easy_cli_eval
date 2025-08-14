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
    """Test OpenAI provider supports GPT-5."""
    # Mock the OpenAI client
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    # Mock the response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "GPT-5 response"
    mock_client.chat.completions.create.return_value = mock_response
    
    # Create the provider with GPT-5
    provider = OpenAIProvider(model="gpt-5", api_key="test_key")
    
    # Call generate
    response = provider.generate("Test prompt")
    
    # Check the response
    assert response == "GPT-5 response"
    
    # Check the client call used the correct model
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args["model"] == "gpt-5"


@patch("src.llm_benchmark.providers.gemini.genai")
def test_gemini_25_pro_support(mock_genai):
    """Test Gemini provider supports Gemini 2.5 Pro."""
    # Mock the Gemini GenerativeModel
    mock_model = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model
    
    # Mock the response
    mock_response = MagicMock()
    mock_response.text = "Gemini 2.5 Pro response"
    mock_model.generate_content.return_value = mock_response
    
    # Mock the genai.configure function to prevent API key validation
    mock_genai.configure = MagicMock()
    
    # Create the provider with Gemini 2.5 Pro
    provider = GeminiProvider(model="gemini-2.5-pro", api_key="test_key")
    
    # Call generate
    response = provider.generate("Test prompt")
    
    # Check the response
    assert response == "Gemini 2.5 Pro response"
    
    # Check the model was created with the correct name
    mock_genai.GenerativeModel.assert_called_once()
    call_args = mock_genai.GenerativeModel.call_args[1]
    assert call_args["model_name"] == "gemini-2.5-pro"


@patch("src.llm_benchmark.providers.grok.OpenAI")
def test_grok_2_support(mock_openai):
    """Test Grok provider supports Grok-2."""
    # Mock the OpenAI client (Grok uses OpenAI-compatible API)
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    # Mock the response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Grok-2 response"
    mock_client.chat.completions.create.return_value = mock_response
    
    # Create the provider with Grok-2
    provider = GrokProvider(model="grok-2", api_key="test_key")
    
    # Call generate
    response = provider.generate("Test prompt")
    
    # Check the response
    assert response == "Grok-2 response"
    
    # Check the client call used the correct model
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args[1]
    assert call_args["model"] == "grok-2"
