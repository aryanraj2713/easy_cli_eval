

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

console = Console()

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
    
    logger.info(
        "compare_start",
        models=models,
        task=task,
        method=method,
        dataset=dataset,
        num_samples=num_samples,
    )
    
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
    
    results = {}
    
    for provider, model in models:
        console.print(f"\n[bold]Benchmarking {provider}:{model}...[/bold]")
        
        try:
            model_results = run_command(
                provider=provider,
                model=model,
                task=task,
                method=method,
                dataset=dataset,
                num_samples=num_samples,
                output=None,
            )
            
            results[f"{provider}:{model}"] = model_results
            
        except Exception as e:
            logger.error(
                "model_benchmark_failed",
                provider=provider,
                model=model,
                error=str(e),
            )
            console.print(f"[bold red]Error with {provider}:{model}:[/bold red] {str(e)}")
    
    if results:
        _display_comparison(results)
        
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
    
    all_metrics = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())
    
    table = Table(title="Model Comparison")
    table.add_column("Metric", style="cyan")
    
    for model in results.keys():
        table.add_column(model, style="green")
    
    for metric in sorted(all_metrics):
        row = [metric]
        
        for model in results.keys():
            value = results[model].get(metric, "N/A")
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            else:
                row.append(str(value))
        
        table.add_row(*row)
    
    console.print(table)
    
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
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix.lower() == ".json":
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    elif output_path.suffix.lower() == ".csv":
        import csv
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            header = ["Metric"] + list(results["results"].keys())
            writer.writerow(header)
            
            all_metrics = set()
            for model_results in results["results"].values():
                all_metrics.update(model_results.keys())
            
            for metric in sorted(all_metrics):
                row = [metric]
                for model in results["results"].keys():
                    value = results["results"][model].get(metric, "")
                    row.append(str(value))
                writer.writerow(row)
    else:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    
    console.print(f"[bold green]Results saved to:[/bold green] {output_path}")
    logger.info("results_saved", output_path=str(output_path))
