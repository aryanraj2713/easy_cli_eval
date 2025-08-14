"""
List models command implementation for LLM Benchmark CLI.

This module contains the implementation of the 'list-models' command.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...core.exceptions import ConfigurationError, ProviderError
from ...providers.factory import ProviderFactory
from ...utils.helpers import timed
from ...utils.logging import get_logger

# Initialize console for rich output
console = Console()

# Initialize logger
logger = get_logger(__name__)


@timed
def list_models_command(provider: str) -> None:
    """
    List available models for a provider.
    
    Args:
        provider: Provider name (openai, gemini, grok)
        
    Raises:
        ConfigurationError: If the provider is invalid
        ProviderError: If there is an error with the provider
    """
    # Log the start of the command
    logger.info("list_models_start", provider=provider)
    
    # Display command information
    console.print(
        Panel(
            f"[bold]Listing models for provider[/bold]\n"
            f"Provider: [cyan]{provider}[/cyan]",
            title="LLM Benchmark",
            expand=False,
        )
    )
    
    try:
        # Create a provider instance with a dummy model
        # We'll use this to get the list of available models
        provider_instance = ProviderFactory.create(provider, model="dummy")
        
        # Get available models
        console.print("[bold]Fetching available models...[/bold]")
        with console.status("[bold green]Fetching...[/bold green]", spinner="dots"):
            models = provider_instance.get_available_models()
        
        # Display models
        if models:
            _display_models(provider, models)
        else:
            console.print("[bold yellow]No models found[/bold yellow]")
        
        # Log successful completion
        logger.info(
            "list_models_complete",
            provider=provider,
            model_count=len(models),
        )
        
    except Exception as e:
        logger.error(
            "list_models_failed",
            provider=provider,
            error=str(e),
        )
        raise


def _display_models(provider: str, models: list) -> None:
    """
    Display available models.
    
    Args:
        provider: Provider name
        models: List of model names
    """
    # Create a table for the models
    table = Table(title=f"Available Models for {provider}")
    table.add_column("Model", style="cyan")
    
    # Add rows for each model
    for model in sorted(models):
        table.add_row(model)
    
    # Display the table
    console.print(table)
