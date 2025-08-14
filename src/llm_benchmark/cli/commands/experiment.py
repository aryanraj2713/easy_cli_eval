

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

console = Console()

logger = get_logger(__name__)

@timed
def experiment_command(
    config_path: Path,
    output_dir: Optional[Path] = None,
) -> None:
    
    logger.info(
        "experiment_start",
        config_path=str(config_path),
        output_dir=str(output_dir) if output_dir else None,
    )
    
    console.print(
        Panel(
            f"[bold]Running experiment[/bold]\n"
            f"Config: [cyan]{config_path}[/cyan]",
            title="LLM Benchmark",
            expand=False,
        )
    )
    
    config = load_config(config_path)
    
    if output_dir:
        config.output.output_dir = str(output_dir)
    
    output_dir = ensure_directory(Path(config.output.output_dir))
    
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
    
    results = _run_experiment(config, output_dir)
    
    _save_overall_results(config, results, output_dir)
    
    logger.info(
        "experiment_complete",
        config_path=str(config_path),
        output_dir=str(output_dir),
    )
    
    console.print(f"\n[bold green]Experiment completed![/bold green]")
    console.print(f"Results saved to: {output_dir}")

def _run_experiment(config, output_dir: Path) -> Dict:
    
    results = {}
    
    total_runs = 0
    if config.methods.traditional:
        total_runs += len(config.methods.traditional.methods) * len(config.models) * len(config.tasks)
    if config.methods.gape:
        total_runs += len(config.models) * len(config.tasks)
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Running experiment...", total=total_runs)
        
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
    
    method_results = {}
    
    for task_config in config.tasks:
        task_name = task_config.name
        dataset = task_config.dataset
        
        task_dir = ensure_directory(output_dir / method / task_name)
        
        console.print(f"\n[bold]Running {method} on {task_name}...[/bold]")
        
        models = [(m.provider, m.model) for m in config.models]
        
        try:
            task_results = compare_command(
                models=models,
                task=task_name,
                method=method,
                dataset=dataset,
                num_samples=10,  
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
        
        progress.update(task_id, advance=len(models))
    
    return method_results

def _save_overall_results(config, results: Dict, output_dir: Path) -> None:
    
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
    
    if "json" in config.output.format:
        pass
    
    if "csv" in config.output.format:
        _generate_csv_summary(config, results, output_dir)
    
    if "html" in config.output.format and config.output.include_plots:
        _generate_html_report(config, results, output_dir)

def _generate_csv_summary(config, results: Dict, output_dir: Path) -> None:
    
    import csv
    
    with open(output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        
        header = ["Method", "Task", "Model", "Metric", "Value"]
        writer.writerow(header)
        
        for method, method_results in results.items():
            for task, task_results in method_results.items():
                for model, model_results in task_results.items():
                    for metric, value in model_results.items():
                        writer.writerow([method, task, model, metric, value])

def _generate_html_report(config, results: Dict, output_dir: Path) -> None:
    
    with open(output_dir / "report.html", "w") as f:
        f.write(
            f"<html><head><title>{config.name}</title></head><body>"
            f"<h1>{config.name}</h1>"
            f"<p>{config.description or ''}</p>"
            f"<p>See results.json for full results</p>"
            f"</body></html>"
        )
