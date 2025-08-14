

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...core.exceptions import ConfigurationError, ProviderError
from ...providers.factory import ProviderFactory
from ...utils.helpers import timed
from ...utils.logging import get_logger

console = Console()

logger = get_logger(__name__)

@timed
def list_models_command(provider: str) -> None:
    
    logger.info("list_models_start", provider=provider)
    
    console.print(
        Panel(
            f"[bold]Listing models for provider[/bold]\n"
            f"Provider: [cyan]{provider}[/cyan]",
            title="LLM Benchmark",
            expand=False,
        )
    )
    
    try:
        provider_instance = ProviderFactory.create(provider, model="dummy")
        
        console.print("[bold]Fetching available models...[/bold]")
        with console.status("[bold green]Fetching...[/bold green]", spinner="dots"):
            models = provider_instance.get_available_models()
        
        if models:
            _display_models(provider, models)
        else:
            console.print("[bold yellow]No models found[/bold yellow]")
        
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
    
    table = Table(title=f"Available Models for {provider}")
    table.add_column("Model", style="cyan")
    
    for model in sorted(models):
        table.add_row(model)
    
    console.print(table)
