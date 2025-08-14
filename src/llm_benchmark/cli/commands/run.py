"""
Run command implementation for LLM Benchmark CLI.

This module contains the implementation of the 'run' command.
"""

import json
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...core.exceptions import ConfigurationError, ProviderError
from ...providers.factory import ProviderFactory
from ...utils.helpers import format_duration, timed
from ...utils.logging import get_logger

# Initialize console for rich output
console = Console()

# Initialize logger
logger = get_logger(__name__)


@timed
def run_command(
    provider: str,
    model: str,
    task: str,
    method: str,
    dataset: Optional[str] = None,
    num_samples: int = 10,
    output: Optional[Path] = None,
) -> Dict:
    """
    Run a benchmark for a specific model on a specific task.
    
    Args:
        provider: Provider name (openai, gemini, grok)
        model: Model name
        task: Task name (qa, summarization, etc.)
        method: Evaluation method (zero_shot, few_shot, chain_of_thought, gape)
        dataset: Dataset name (defaults to a standard dataset for the task)
        num_samples: Number of samples to evaluate
        output: Output file path
        
    Returns:
        Dict with benchmark results
        
    Raises:
        ConfigurationError: If the configuration is invalid
        ProviderError: If there is an error with the provider
    """
    # Log the start of the benchmark
    logger.info(
        "benchmark_start",
        provider=provider,
        model=model,
        task=task,
        method=method,
        dataset=dataset,
        num_samples=num_samples,
    )
    
    # Display benchmark information
    console.print(
        Panel(
            f"[bold]Running benchmark[/bold]\n"
            f"Provider: [cyan]{provider}[/cyan]\n"
            f"Model: [cyan]{model}[/cyan]\n"
            f"Task: [cyan]{task}[/cyan]\n"
            f"Method: [cyan]{method}[/cyan]\n"
            f"Dataset: [cyan]{dataset or 'default'}[/cyan]\n"
            f"Samples: [cyan]{num_samples}[/cyan]",
            title="LLM Benchmark",
            expand=False,
        )
    )
    
    # Determine dataset if not provided
    if dataset is None:
        dataset = _get_default_dataset(task)
        logger.info("using_default_dataset", task=task, dataset=dataset)
    
    try:
        # Create provider instance
        provider_instance = ProviderFactory.create(provider, model)
        
        # Create evaluator based on method
        evaluator = _create_evaluator(method, provider_instance)
        
        # Run evaluation
        console.print("[bold]Running evaluation...[/bold]")
        with console.status("[bold green]Evaluating...[/bold green]", spinner="dots"):
            results = evaluator.evaluate(
                model=provider_instance,
                dataset=dataset,
                config={
                    "task": task,
                    "num_samples": num_samples,
                },
            )
        
        # Display results
        _display_results(results)
        
        # Save results if output path provided
        if output:
            _save_results(
                output,
                {
                    "provider": provider,
                    "model": model,
                    "task": task,
                    "method": method,
                    "dataset": dataset,
                    "num_samples": num_samples,
                    "results": results,
                },
            )
        
        # Log successful completion
        logger.info(
            "benchmark_complete",
            provider=provider,
            model=model,
            task=task,
            method=method,
            dataset=dataset,
            num_samples=num_samples,
            results=results,
        )
        
        return results
        
    except Exception as e:
        logger.error(
            "benchmark_failed",
            provider=provider,
            model=model,
            task=task,
            method=method,
            dataset=dataset,
            error=str(e),
        )
        raise


def _get_default_dataset(task: str) -> str:
    """
    Get the default dataset for a task.
    
    Args:
        task: Task name
        
    Returns:
        Dataset name
    """
    # Map tasks to default datasets
    task_datasets = {
        "qa": "squad_v2",
        "summarization": "cnn_dailymail",
        "translation": "wmt16",
        "classification": "glue",
    }
    
    return task_datasets.get(task, "default")


def _create_evaluator(method: str, provider_instance):
    """
    Create an evaluator based on the method.
    
    Args:
        method: Evaluation method
        provider_instance: Provider instance
        
    Returns:
        Evaluator instance
        
    Raises:
        ConfigurationError: If the method is invalid
    """
    if method in ["zero_shot", "few_shot", "chain_of_thought"]:
        from ...evaluation.traditional import TraditionalEvaluator
        return TraditionalEvaluator(provider=provider_instance, method=method)
    elif method == "gape":
        # GAPE is implemented directly in the provider
        # We'll create a simple wrapper evaluator
        from ...evaluation.base import BaseEvaluator
        
        class GAPEEvaluator(BaseEvaluator):
            def evaluate(self, model, dataset, config):
                result = model.gape(
                    base_prompt=f"Task: {config['task']}\nDataset: {dataset}",
                    target_task=config['task'],
                )
                return result['final_metrics']
            
            def get_required_config(self):
                return ["task"]
        
        return GAPEEvaluator(provider=provider_instance)
    else:
        raise ConfigurationError(f"Invalid method: {method}")


def _display_results(results: Dict) -> None:
    """
    Display benchmark results.
    
    Args:
        results: Benchmark results
    """
    # Create a table for the results
    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Add rows for each metric
    for metric, value in results.items():
        table.add_row(metric, f"{value:.4f}")
    
    # Display the table
    console.print(table)


def _save_results(output_path: Path, results: Dict) -> None:
    """
    Save benchmark results to a file.
    
    Args:
        output_path: Output file path
        results: Benchmark results
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
            writer.writerow(["Metric", "Value"])
            for metric, value in results["results"].items():
                writer.writerow([metric, value])
    else:
        # Default to JSON
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    
    console.print(f"[bold green]Results saved to:[/bold green] {output_path}")
    logger.info("results_saved", output_path=str(output_path))
