"""
Configuration utilities for LLM Benchmark CLI.

This module contains utilities for loading and validating configuration files.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import ValidationError

from ..core.exceptions import ConfigurationError
from ..core.models import ExperimentConfig


def load_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """
    Load and validate a configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Validated experiment configuration
        
    Raises:
        ConfigurationError: If the configuration is invalid or the file cannot be loaded
    """
    try:
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        # Determine file format from extension
        if config_path.suffix.lower() == ".yaml" or config_path.suffix.lower() == ".yml":
            config_dict = _load_yaml(config_path)
        elif config_path.suffix.lower() == ".toml":
            config_dict = _load_toml(config_path)
        elif config_path.suffix.lower() == ".json":
            config_dict = _load_json(config_path)
        else:
            raise ConfigurationError(
                f"Unsupported configuration file format: {config_path.suffix}"
            )
        
        # Validate with Pydantic model
        try:
            config = ExperimentConfig.model_validate(config_dict)
            return config
        except ValidationError as e:
            raise ConfigurationError(f"Invalid configuration: {str(e)}")
            
    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        raise ConfigurationError(f"Failed to load configuration: {str(e)}")


def validate_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate a configuration file without loading it fully.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict with validation results
        
    Raises:
        ConfigurationError: If the file cannot be loaded
    """
    try:
        # Try to load the config
        config = load_config(config_path)
        
        # If we get here, the config is valid
        return {
            "valid": True,
            "models": [f"{m.provider}:{m.model}" for m in config.models],
            "tasks": [t.name for t in config.tasks],
            "methods": {
                "traditional": (
                    config.methods.traditional.methods if config.methods.traditional else []
                ),
                "gape": bool(config.methods.gape),
            },
        }
    except ValidationError as e:
        # Return validation errors
        return {
            "valid": False,
            "errors": str(e),
        }
    except Exception as e:
        raise ConfigurationError(f"Failed to validate configuration: {str(e)}")


def _load_yaml(file_path: Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Dict with configuration values
        
    Raises:
        ConfigurationError: If the file cannot be loaded
    """
    try:
        import yaml
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except ImportError:
        raise ConfigurationError("PyYAML is required for YAML configuration files")
    except Exception as e:
        raise ConfigurationError(f"Failed to load YAML configuration: {str(e)}")


def _load_toml(file_path: Path) -> Dict[str, Any]:
    """
    Load a TOML configuration file.
    
    Args:
        file_path: Path to the TOML file
        
    Returns:
        Dict with configuration values
        
    Raises:
        ConfigurationError: If the file cannot be loaded
    """
    try:
        # Use tomli for Python < 3.11, tomllib for Python >= 3.11
        if hasattr(json, "loads"):  # Python >= 3.11
            import tomllib
            with open(file_path, "rb") as f:
                return tomllib.load(f)
        else:
            import tomli
            with open(file_path, "rb") as f:
                return tomli.load(f)
    except ImportError:
        raise ConfigurationError("tomli is required for TOML configuration files")
    except Exception as e:
        raise ConfigurationError(f"Failed to load TOML configuration: {str(e)}")


def _load_json(file_path: Path) -> Dict[str, Any]:
    """
    Load a JSON configuration file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dict with configuration values
        
    Raises:
        ConfigurationError: If the file cannot be loaded
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise ConfigurationError(f"Failed to load JSON configuration: {str(e)}")


def get_default_config_path() -> Path:
    """
    Get the default configuration directory path.
    
    Returns:
        Path to the default configuration directory
    """
    # Check for config in the current directory
    cwd_config = Path.cwd() / "configs"
    if cwd_config.exists() and cwd_config.is_dir():
        return cwd_config
    
    # Check for config in the package directory
    package_dir = Path(__file__).parent.parent.parent.parent
    package_config = package_dir / "configs"
    if package_config.exists() and package_config.is_dir():
        return package_config
    
    # Fall back to the current directory
    return Path.cwd()
