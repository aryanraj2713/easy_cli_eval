"""
Experiment command implementation for LLM Benchmark CLI.

This module contains the implementation of the 'experiment' command.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TaskID

from ...core.exceptions import ConfigurationError
from ...utils.config import load_config
from ...utils.helpers import ensure_directory, format_duration, timed
from ...utils.logging import get_logger
from .compare import compare_command
from .run import run_command

# Initialize console for rich output
console = Console()

# Initialize logger
logger = get_logger(__name__)


@timed
def experiment_command(
    config_path: Path,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Run an experiment defined in a configuration file.
    
    Args:
        config_path: Path to the experiment configuration file
        output_dir: Output directory for results (overrides config)
        
    Raises:
        ConfigurationError: If the configuration is invalid
    """
    # Log the start of the experiment
    logger.info(
        "experiment_start",
        config_path=str(config_path),
        output_dir=str(output_dir) if output_dir else None,
    )
    
    # Display experiment information
    console.print(
        Panel(
            f"[bold]Running experiment[/bold]\n"
            f"Config: [cyan]{config_path}[/cyan]",
            title="LLM Benchmark",
            expand=False,
        )
    )
    
    # Load the configuration
    config = load_config(config_path)
    
    # Override output directory if provided
    if output_dir:
        config.output.output_dir = str(output_dir)
    
    # Create output directory
    output_dir = ensure_directory(Path(config.output.output_dir))
    
    # Display experiment details
    console.print(f"\n[bold]Experiment:[/bold] {config.name}")
    if config.description:
        console.print(f"[bold]Description:[/bold] {config.description}")
    
    console.print(f"\n[bold]Models:[/bold] {len(config.models)}")
    for model in config.models:
        console.print(f"  • {model.provider}:{model.model}")
    
    console.print(f"\n[bold]Tasks:[/bold] {len(config.tasks)}")
    for task in config.tasks:
        console.print(f"  • {task.name} ({task.dataset})")
    
    console.print("\n[bold]Methods:[/bold]")
    if config.methods.traditional:
        console.print("  [bold]Traditional:[/bold]")
        for method in config.methods.traditional.methods:
            console.print(f"    • {method}")
    
    if config.methods.gape:
        console.print(
            "  [bold]GAPE:[/bold] "
            f"pop={config.methods.gape.population_size}, "
            f"gen={config.methods.gape.generations}, "
            f"mut={config.methods.gape.mutation_rate}"
        )
    
    # Run the experiment
    results = _run_experiment(config, output_dir)
    
    # Save overall results
    _save_overall_results(config, results, output_dir)
    
    # Log successful completion
    logger.info(
        "experiment_complete",
        config_path=str(config_path),
        output_dir=str(output_dir),
    )
    
    console.print(f"\n[bold green]Experiment completed![/bold green]")
    console.print(f"Results saved to: {output_dir}")


def _run_experiment(config, output_dir: Path) -> Dict:
    """
    Run the experiment according to the configuration.
    
    Args:
        config: Experiment configuration
        output_dir: Output directory for results
        
    Returns:
        Dict with experiment results
    """
    results = {}
    
    # Set up progress tracking
    total_runs = 0
    if config.methods.traditional:
        total_runs += len(config.methods.traditional.methods) * len(config.models) * len(config.tasks)
    if config.methods.gape:
        total_runs += len(config.models) * len(config.tasks)
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Running experiment...", total=total_runs)
        
        # Run traditional methods
        if config.methods.traditional:
            for method in config.methods.traditional.methods:
                method_results = _run_method(
                    config=config,
                    method=method,
                    output_dir=output_dir,
                    progress=progress,
                    task_id=task,
                )
                results[method] = method_results
        
        # Run GAPE
        if config.methods.gape:
            gape_results = _run_method(
                config=config,
                method="gape",
                output_dir=output_dir,
                progress=progress,
                task_id=task,
            )
            results["gape"] = gape_results
    
    return results


def _run_method(
    config,
    method: str,
    output_dir: Path,
    progress: Progress,
    task_id: TaskID,
) -> Dict:
    """
    Run a specific method across all models and tasks.
    
    Args:
        config: Experiment configuration
        method: Method name
        output_dir: Output directory for results
        progress: Progress tracker
        task_id: Task ID for progress tracking
        
    Returns:
        Dict with method results
    """
    method_results = {}
    
    for task_config in config.tasks:
        task_name = task_config.name
        dataset = task_config.dataset
        
        # Create task directory
        task_dir = ensure_directory(output_dir / method / task_name)
        
        # Compare all models on this task
        console.print(f"\n[bold]Running {method} on {task_name}...[/bold]")
        
        # Prepare models for comparison
        models = [(m.provider, m.model) for m in config.models]
        
        # Run the comparison
        try:
            task_results = compare_command(
                models=models,
                task=task_name,
                method=method,
                dataset=dataset,
                num_samples=10,  # TODO: Make configurable
                output=task_dir / "results.json",
            )
            
            method_results[task_name] = task_results
            
        except Exception as e:
            logger.error(
                "task_failed",
                method=method,
                task=task_name,
                error=str(e),
            )
            console.print(f"[bold red]Error with {method} on {task_name}:[/bold red] {str(e)}")
        
        # Update progress
        progress.update(task_id, advance=len(models))
    
    return method_results


def _save_overall_results(config, results: Dict, output_dir: Path) -> None:
    """
    Save overall experiment results.
    
    Args:
        config: Experiment configuration
        results: Experiment results
        output_dir: Output directory for results
    """
    # Save overall results as JSON
    with open(output_dir / "results.json", "w") as f:
        json.dump(
            {
                "experiment": {
                    "name": config.name,
                    "description": config.description,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    
    # Generate summary if requested
    if "json" in config.output.format:
        # Already saved above
        pass
    
    if "csv" in config.output.format:
        _generate_csv_summary(config, results, output_dir)
    
    if "html" in config.output.format and config.output.include_plots:
        _generate_html_report(config, results, output_dir)


def _generate_csv_summary(config, results: Dict, output_dir: Path) -> None:
    """
    Generate CSV summary of experiment results.
    
    Args:
        config: Experiment configuration
        results: Experiment results
        output_dir: Output directory for results
    """
    import csv
    
    # Create CSV file
    with open(output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write header
        header = ["Method", "Task", "Model", "Metric", "Value"]
        writer.writerow(header)
        
        # Write rows
        for method, method_results in results.items():
            for task, task_results in method_results.items():
                for model, model_results in task_results.items():
                    for metric, value in model_results.items():
                        writer.writerow([method, task, model, metric, value])


def _generate_html_report(config, results: Dict, output_dir: Path) -> None:
    """
    Generate HTML report with plots.
    
    Args:
        config: Experiment configuration
        results: Experiment results
        output_dir: Output directory for results
    """
    # In a real implementation, this would generate an HTML report with plots
    # For now, we'll just create a placeholder file
    with open(output_dir / "report.html", "w") as f:
        f.write(
            f"<html><head><title>{config.name}</title></head><body>"
            f"<h1>{config.name}</h1>"
            f"<p>{config.description or ''}</p>"
            f"<p>See results.json for full results</p>"
            f"</body></html>"
        )
