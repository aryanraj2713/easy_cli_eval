"""
Tests for configuration handling.
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.llm_benchmark.core.exceptions import ConfigurationError
from src.llm_benchmark.utils.config import load_config, validate_config


def test_load_yaml_config():
    """Test loading a YAML configuration file."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+") as f:
        f.write("""
name: "Test Experiment"
description: "Test experiment description"
  
models:
  - provider: openai
    model: gpt-4
  - provider: gemini
    model: gemini-pro

tasks:
  - name: question_answering
    dataset: squad_v2
    metrics: [accuracy, f1_score]

methods:
  traditional:
    methods:
      - zero_shot
  
  gepa:
    population_size: 10
    generations: 5
    mutation_rate: 0.3
    fitness_function: composite_score

output:
  format: [json]
  include_plots: false
  save_intermediate: false
        """)
        f.flush()
        
        config = load_config(f.name)
        
        assert config.name == "Test Experiment"
        assert config.description == "Test experiment description"
        assert len(config.models) == 2
        assert config.models[0].provider == "openai"
        assert config.models[0].model == "gpt-4"
        assert config.models[1].provider == "gemini"
        assert config.models[1].model == "gemini-pro"
        assert len(config.tasks) == 1
        assert config.tasks[0].name == "question_answering"
        assert config.tasks[0].dataset == "squad_v2"
        assert config.tasks[0].metrics == ["accuracy", "f1_score"]
        assert config.methods.traditional.methods == ["zero_shot"]
        assert config.methods.gepa.population_size == 10
        assert config.methods.gepa.generations == 5
        assert config.methods.gepa.mutation_rate == 0.3
        assert config.methods.gepa.fitness_function == "composite_score"
        assert config.output.format == ["json"]
        assert config.output.include_plots is False
        assert config.output.save_intermediate is False


def test_validate_valid_config():
    """Test validating a valid configuration file."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+") as f:
        f.write("""
name: "Test Experiment"
  
models:
  - provider: openai
    model: gpt-4

tasks:
  - name: question_answering
    dataset: squad_v2
    metrics: [accuracy]

methods:
  traditional:
    methods:
      - zero_shot
        """)
        f.flush()
        
        result = validate_config(f.name)
        
        assert result["valid"] is True
        assert result["models"] == ["openai:gpt-4"]
        assert result["tasks"] == ["question_answering"]
        assert result["methods"]["traditional"] == ["zero_shot"]
        assert result["methods"]["gepa"] is False


def test_load_invalid_config():
    """Test loading an invalid configuration file."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+") as f:
        f.write("""
name: "Test Experiment"
  
models:
  - provider: openai
    # Missing model field

tasks:
  - name: question_answering
    dataset: squad_v2

methods:
  traditional:
    methods:
      - zero_shot
        """)
        f.flush()
        
        with pytest.raises(ConfigurationError):
            load_config(f.name)


def test_load_nonexistent_config():
    """Test loading a nonexistent configuration file."""
    with pytest.raises(ConfigurationError):
        load_config("/nonexistent/path/config.yaml")