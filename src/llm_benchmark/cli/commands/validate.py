"""
Validate command implementation for LLM Benchmark CLI.

This module contains the implementation of the 'validate-config' command.
"""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from ...utils.config import validate_config
from ...utils.logging import get_logger

# Initialize console for rich output
console = Console()

# Initialize logger
logger = get_logger(__name__)


def validate_command(config_path: Path) -> None:
    """
    Validate a configuration file.
    
    Args:
        config_path: Path to the configuration file
    """
    # Log the start of validation
    logger.info("validate_start", config_path=str(config_path))
    
    # Display validation information
    console.print(
        Panel(
            f"[bold]Validating configuration file[/bold]\n"
            f"Path: [cyan]{config_path}[/cyan]",
            title="LLM Benchmark",
            expand=False,
        )
    )
    
    # Validate the configuration
    try:
        result = validate_config(config_path)
        
        if result["valid"]:
            # Display success message
            console.print("[bold green]✓ Configuration is valid[/bold green]")
            
            # Display configuration details
            console.print("\n[bold]Configuration details:[/bold]")
            
            # Models
            console.print("[bold cyan]Models:[/bold cyan]")
            for model in result["models"]:
                console.print(f"  • {model}")
            
            # Tasks
            console.print("\n[bold cyan]Tasks:[/bold cyan]")
            for task in result["tasks"]:
                console.print(f"  • {task}")
            
            # Methods
            console.print("\n[bold cyan]Methods:[/bold cyan]")
            
            # Traditional methods
            if result["methods"]["traditional"]:
                console.print("  [bold]Traditional:[/bold]")
                for method in result["methods"]["traditional"]:
                    console.print(f"    • {method}")
            
            # GAPE
            if result["methods"]["gape"]:
                console.print("  [bold]GAPE:[/bold] Enabled")
            
            # Log successful validation
            logger.info(
                "validate_complete",
                config_path=str(config_path),
                valid=True,
                models=result["models"],
                tasks=result["tasks"],
            )
        else:
            # Display error message
            console.print("[bold red]✗ Configuration is invalid[/bold red]")
            console.print(f"\n[bold]Errors:[/bold]\n{result['errors']}")
            
            # Log validation failure
            logger.error(
                "validate_failed",
                config_path=str(config_path),
                errors=result["errors"],
            )
    except Exception as e:
        # Display error message
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        
        # Log error
        logger.error(
            "validate_error",
            config_path=str(config_path),
            error=str(e),
        )
