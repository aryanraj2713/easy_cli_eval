

import os
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from .. import __version__, providers
from ..utils.config import load_config, validate_config
from ..utils.logging import configure_logging, get_logger

app = typer.Typer(
    name="llm-benchmark",
    help="CLI tool to benchmark Large Language Models with modular providers and evaluation methods",
    add_completion=False,
)

console = Console()

logger = get_logger(__name__)

def version_callback(value: bool) -> None:
    
    if value:
        console.print(f"LLM Benchmark CLI v{__version__}")
        raise typer.Exit()

@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-v", callback=version_callback, help="Show version and exit"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l", help="Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    ),
) -> None:
    
    configure_logging(log_level)

@app.command("run")
def run_benchmark(
    provider: str = typer.Option(..., "--provider", "-p", help="Provider name (openai, gemini, grok)"),
    model: str = typer.Option(..., "--model", "-m", help="Model name"),
    task: str = typer.Option(..., "--task", "-t", help="Task name (qa, summarization, etc.)"),
    method: str = typer.Option(
        "dspy", "--method", "-e", 
        help="Evaluation method (dspy, zero_shot, few_shot, chain_of_thought, gepa)"
    ),
    dataset: Optional[str] = typer.Option(
        None, "--dataset", "-d", help="Dataset name (defaults to a standard dataset for the task)"
    ),
    num_samples: int = typer.Option(
        10, "--num-samples", "-n", help="Number of samples to evaluate"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
) -> None:
    
    from .commands.run import run_command
    
    try:
        run_command(
            provider=provider,
            model=model,
            task=task,
            method=method,
            dataset=dataset,
            num_samples=num_samples,
            output=output,
        )
    except Exception as e:
        logger.error("benchmark_failed", error=str(e))
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)

@app.command("compare")
def compare_models(
    models: str = typer.Option(
        ..., "--models", "-m", 
        help="Comma-separated list of models to compare (format: provider:model)"
    ),
    task: str = typer.Option(..., "--task", "-t", help="Task name (qa, summarization, etc.)"),
    method: str = typer.Option(
        "dspy", "--method", "-e", 
        help="Evaluation method (dspy, zero_shot, few_shot, chain_of_thought, gepa)"
    ),
    dataset: Optional[str] = typer.Option(
        None, "--dataset", "-d", help="Dataset name (defaults to a standard dataset for the task)"
    ),
    num_samples: int = typer.Option(
        10, "--num-samples", "-n", help="Number of samples to evaluate"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
) -> None:
    
    from .commands.compare import compare_command
    
    try:
        model_specs = []
        for model_spec in models.split(","):
            if ":" not in model_spec:
                raise ValueError(
                    f"Invalid model specification: {model_spec}. "
                    f"Format should be provider:model (e.g., openai:gpt-4)"
                )
            provider, model = model_spec.split(":", 1)
            model_specs.append((provider, model))
        
        compare_command(
            models=model_specs,
            task=task,
            method=method,
            dataset=dataset,
            num_samples=num_samples,
            output=output,
        )
    except Exception as e:
        logger.error("compare_failed", error=str(e))
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)

@app.command("experiment")
def run_experiment(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to experiment configuration file"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Output directory for results"
    ),
) -> None:
    
    from .commands.experiment import experiment_command
    
    try:
        experiment_command(
            config_path=config,
            output_dir=output_dir,
        )
    except Exception as e:
        logger.error("experiment_failed", error=str(e))
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)

@app.command("list-models")
def list_models(
    provider: str = typer.Option(..., "--provider", "-p", help="Provider name (openai, gemini, grok)"),
) -> None:
    
    from .commands.list_models import list_models_command
    
    try:
        list_models_command(provider=provider)
    except Exception as e:
        logger.error("list_models_failed", error=str(e))
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)

@app.command("validate-config")
def validate_config_cmd(
    file: Path = typer.Option(..., "--file", "-f", help="Path to configuration file"),
) -> None:
    
    from .commands.validate import validate_command
    
    try:
        validate_command(config_path=file)
    except Exception as e:
        logger.error("validate_failed", error=str(e))
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
