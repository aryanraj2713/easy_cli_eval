"""
Compare command implementation for LLM Benchmark CLI.

This module contains the implementation of the 'compare' command.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...core.exceptions import ConfigurationError, ProviderError
from ...providers.factory import ProviderFactory
from ...utils.helpers import format_duration, timed
from ...utils.logging import get_logger
from .run import run_command

# Initialize console for rich output
console = Console()

# Initialize logger
logger = get_logger(__name__)


@timed
def compare_command(
    models: List[Tuple[str, str]],
    task: str,
    method: str,
    dataset: Optional[str] = None,
    num_samples: int = 10,
    output: Optional[Path] = None,
) -> Dict:
    """
    Compare multiple models on a specific task.
    
    Args:
        models: List of (provider, model) tuples
        task: Task name (qa, summarization, etc.)
        method: Evaluation method (zero_shot, few_shot, chain_of_thought, gape)
        dataset: Dataset name (defaults to a standard dataset for the task)
        num_samples: Number of samples to evaluate
        output: Output file path
        
    Returns:
        Dict with comparison results
        
    Raises:
        ConfigurationError: If the configuration is invalid
        ProviderError: If there is an error with a provider
    """
    # Log the start of the comparison
    logger.info(
        "compare_start",
        models=models,
        task=task,
        method=method,
        dataset=dataset,
        num_samples=num_samples,
    )
    
    # Display comparison information
    model_strings = [f"{provider}:{model}" for provider, model in models]
    console.print(
        Panel(
            f"[bold]Comparing models[/bold]\n"
            f"Models: [cyan]{', '.join(model_strings)}[/cyan]\n"
            f"Task: [cyan]{task}[/cyan]\n"
            f"Method: [cyan]{method}[/cyan]\n"
            f"Dataset: [cyan]{dataset or 'default'}[/cyan]\n"
            f"Samples: [cyan]{num_samples}[/cyan]",
            title="LLM Benchmark",
            expand=False,
        )
    )
    
    # Run benchmarks for each model
    results = {}
    
    for provider, model in models:
        console.print(f"\n[bold]Benchmarking {provider}:{model}...[/bold]")
        
        try:
            # Run the benchmark
            model_results = run_command(
                provider=provider,
                model=model,
                task=task,
                method=method,
                dataset=dataset,
                num_samples=num_samples,
                # Don't save individual results
                output=None,
            )
            
            # Store the results
            results[f"{provider}:{model}"] = model_results
            
        except Exception as e:
            logger.error(
                "model_benchmark_failed",
                provider=provider,
                model=model,
                error=str(e),
            )
            console.print(f"[bold red]Error with {provider}:{model}:[/bold red] {str(e)}")
            # Continue with other models
    
    # Display comparison results
    if results:
        _display_comparison(results)
        
        # Save results if output path provided
        if output:
            _save_results(
                output,
                {
                    "task": task,
                    "method": method,
                    "dataset": dataset,
                    "num_samples": num_samples,
                    "results": results,
                },
            )
    else:
        console.print("[bold red]No results to compare[/bold red]")
    
    # Log successful completion
    logger.info(
        "compare_complete",
        models=models,
        task=task,
        method=method,
        dataset=dataset,
        num_samples=num_samples,
        results=results,
    )
    
    return results


def _display_comparison(results: Dict[str, Dict[str, float]]) -> None:
    """
    Display comparison results.
    
    Args:
        results: Dict mapping model names to benchmark results
    """
    # Get all metrics
    all_metrics = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())
    
    # Create a table for the results
    table = Table(title="Model Comparison")
    table.add_column("Metric", style="cyan")
    
    # Add columns for each model
    for model in results.keys():
        table.add_column(model, style="green")
    
    # Add rows for each metric
    for metric in sorted(all_metrics):
        row = [metric]
        
        for model in results.keys():
            value = results[model].get(metric, "N/A")
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            else:
                row.append(str(value))
        
        table.add_row(*row)
    
    # Display the table
    console.print(table)
    
    # Determine best model for each metric
    console.print("\n[bold]Best model per metric:[/bold]")
    for metric in sorted(all_metrics):
        best_model = None
        best_value = -float("inf")
        
        for model, model_results in results.items():
            if metric in model_results and isinstance(model_results[metric], (int, float)):
                if model_results[metric] > best_value:
                    best_value = model_results[metric]
                    best_model = model
        
        if best_model:
            console.print(
                f"  • {metric}: [bold green]{best_model}[/bold green] "
                f"([cyan]{best_value:.4f}[/cyan])"
            )


def _save_results(output_path: Path, results: Dict) -> None:
    """
    Save comparison results to a file.
    
    Args:
        output_path: Output file path
        results: Comparison results
    """
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine output format based on file extension
    if output_path.suffix.lower() == ".json":
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    elif output_path.suffix.lower() == ".csv":
        import csv
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Write header
            header = ["Metric"] + list(results["results"].keys())
            writer.writerow(header)
            
            # Get all metrics
            all_metrics = set()
            for model_results in results["results"].values():
                all_metrics.update(model_results.keys())
            
            # Write rows
            for metric in sorted(all_metrics):
                row = [metric]
                for model in results["results"].keys():
                    value = results["results"][model].get(metric, "")
                    row.append(str(value))
                writer.writerow(row)
    else:
        # Default to JSON
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    
    console.print(f"[bold green]Results saved to:[/bold green] {output_path}")
    logger.info("results_saved", output_path=str(output_path))
