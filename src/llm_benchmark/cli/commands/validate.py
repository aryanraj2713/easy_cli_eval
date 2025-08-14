

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from ...utils.config import validate_config
from ...utils.logging import get_logger

console = Console()

logger = get_logger(__name__)

def validate_command(config_path: Path) -> None:
    
    logger.info("validate_start", config_path=str(config_path))
    
    console.print(
        Panel(
            f"[bold]Validating configuration file[/bold]\n"
            f"Path: [cyan]{config_path}[/cyan]",
            title="LLM Benchmark",
            expand=False,
        )
    )
    
    try:
        result = validate_config(config_path)
        
        if result["valid"]:
            console.print("[bold green]✓ Configuration is valid[/bold green]")
            
            console.print("\n[bold]Configuration details:[/bold]")
            
            console.print("[bold cyan]Models:[/bold cyan]")
            for model in result["models"]:
                console.print(f"  • {model}")
            
            console.print("\n[bold cyan]Tasks:[/bold cyan]")
            for task in result["tasks"]:
                console.print(f"  • {task}")
            
            console.print("\n[bold cyan]Methods:[/bold cyan]")
            
            if result["methods"]["traditional"]:
                console.print("  [bold]Traditional:[/bold]")
                for method in result["methods"]["traditional"]:
                    console.print(f"    • {method}")
            
            if result["methods"]["gepa"]:
                console.print("  [bold]GAPE:[/bold] Enabled")
            
            logger.info(
                "validate_complete",
                config_path=str(config_path),
                valid=True,
                models=result["models"],
                tasks=result["tasks"],
            )
        else:
            console.print("[bold red]✗ Configuration is invalid[/bold red]")
            console.print(f"\n[bold]Errors:[/bold]\n{result['errors']}")
            
            logger.error(
                "validate_failed",
                config_path=str(config_path),
                errors=result["errors"],
            )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        
        logger.error(
            "validate_error",
            config_path=str(config_path),
            error=str(e),
        )
