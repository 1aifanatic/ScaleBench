"""CLI entrypoint: loads config, runs experiments."""
import argparse
import sys
import logging
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from core.config import load_config
from evaluation.runner import run_experiment
from evaluation.aggregation import aggregate_from_jsonl, save_summary_csv, save_summary_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run benchmarking experiments for ScaleDown context compression."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments.yaml"),
        help="Path to experiments.yaml config file (default: config/experiments.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and validate without running actual experiments.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Run only the specified experiment ID (optional).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print(f"\n[bold cyan]Loading config from {args.config}[/bold cyan]")

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError, Exception) as e:
        console.print(f"[bold red]Failed to load config:[/bold red] {e}")
        sys.exit(1)

    console.print(f"[green]Loaded {len(config.experiments)} experiment(s)[/green]\n")

    # Filter experiments if --experiment-id is specified
    experiments_to_run = config.experiments
    if args.experiment_id:
        experiments_to_run = [
            exp for exp in config.experiments if exp.id == args.experiment_id
        ]
        if not experiments_to_run:
            console.print(
                f"[bold red]Experiment '{args.experiment_id}' not found in config.[/bold red]"
            )
            console.print(f"Available: {[e.id for e in config.experiments]}")
            sys.exit(1)
        console.print(f"[yellow]Filtered to experiment: {args.experiment_id}[/yellow]\n")

    # Display experiment summary
    _display_experiment_summary(experiments_to_run)

    if args.dry_run:
        console.print("\n[bold yellow]DRY RUN:[/bold yellow] Config validated successfully. No experiments executed.")
        sys.exit(0)

    # Run experiments
    results = []
    failed_experiments = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for exp in experiments_to_run:
            task = progress.add_task(f"Running {exp.id}...", total=None)
            
            try:
                # Run experiment (returns aggregated results directly for multi-task)
                result = run_experiment(exp)
                
                # Check if this is a multi-task experiment
                is_multi_task = result.get("is_multi_task", False)
                
                if is_multi_task:
                    # Multi-task: result already contains aggregated data
                    progress.update(task, description=f"[green]✓[/green] {exp.id} (multi-task)")
                    results.append(result)
                    
                    # Extract timestamp from first raw file
                    first_raw_file = Path(result["raw_output_files"][0])
                    timestamp = first_raw_file.stem.rsplit("_", 1)[-1]
                    
                    # Save summaries
                    summary_dir = Path("results/summaries")
                    save_summary_csv(result, summary_dir / f"{exp.id}_{timestamp}_summary.csv")
                    save_summary_json(result, summary_dir / f"{exp.id}_{timestamp}_summary.json")
                else:
                    # Single-task: need to aggregate from JSONL
                    progress.update(task, description=f"[green]✓[/green] {exp.id}")
                    
                    raw_jsonl = Path(result["raw_output_file"])
                    aggregated = aggregate_from_jsonl(raw_jsonl)
                    results.append(aggregated)
                    
                    # Extract timestamp from filename (format: {exp_id}_{timestamp}.jsonl)
                    timestamp = raw_jsonl.stem.rsplit("_", 1)[-1]
                    
                    summary_dir = Path("results/summaries")
                    save_summary_csv(aggregated, summary_dir / f"{exp.id}_{timestamp}_summary.csv")
                    save_summary_json(aggregated, summary_dir / f"{exp.id}_{timestamp}_summary.json")
                
            except Exception as e:
                progress.update(task, description=f"[red]✗[/red] {exp.id}")
                failed_experiments.append((exp.id, str(e)))
                logger.error(f"Experiment {exp.id} failed: {e}", exc_info=args.verbose)

    # Display final summary
    console.print("\n")
    _display_results_summary(results, failed_experiments)

    # Exit with error if any experiments failed
    if failed_experiments:
        sys.exit(1)


def _display_experiment_summary(experiments):
    """Display summary table of experiments to run."""
    table = Table(title="Experiments to Run", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Dataset", style="yellow")
    table.add_column("Task", style="blue")
    table.add_column("Dataset Task", style="magenta")  # New column for LongBench task
    table.add_column("N", justify="right", style="white")
    
    for exp in experiments:
        # Format dataset task info
        dataset_task = ""
        if exp.dataset.name == "longbench":
            dataset_task = exp.dataset.task or "all (multi-task)"
        
        table.add_row(
            exp.id,
            f"{exp.model.name} ({exp.model.mode})",
            f"{exp.dataset.name} ({exp.dataset.split})",
            exp.task.name,
            dataset_task,
            str(exp.dataset.num_examples or "all"),
        )
    
    console.print(table)
    console.print()


def _display_results_summary(results, failed_experiments):
    """Display final results summary table."""
    if results:
        # Check if any multi-task results
        has_multi_task = any(r.get("is_multi_task", False) for r in results)
        
        # Display main results table
        table = Table(title="Results Summary", show_header=True, header_style="bold green")
        table.add_column("Experiment", style="cyan")
        table.add_column("N", justify="right", style="white")
        table.add_column("Metrics", style="yellow")
        table.add_column("Avg Latency (ms)", justify="right", style="magenta")
        table.add_column("Avg Tokens", justify="right", style="blue")
        table.add_column("Avg Cost ($)", justify="right", style="green")
        
        for result in results:
            # Extract metric means from nested structure
            metrics_str = ", ".join([f"{k}={v['mean']:.2f}" if isinstance(v, dict) else f"{k}={v:.2f}" 
                                     for k, v in result["metrics"].items()])
            
            # Handle both old and new token structure
            if "tokens" in result:
                total_tokens = result["tokens"]["avg_input_tokens"] + result["tokens"]["avg_output_tokens"]
            else:
                total_tokens = result.get("avg_input_tokens", 0) + result.get("avg_output_tokens", 0)
            
            # Handle latency
            if "latency_ms" in result:
                avg_latency = result["latency_ms"]["mean"]
            else:
                avg_latency = result.get("avg_latency_ms", 0)
            
            # Format cost (handle None case)
            if "cost" in result and result["cost"]:
                cost_str = f"${result['cost']['mean_usd']:.4f}"
            elif result.get("avg_cost_usd") is not None:
                cost_str = f"${result['avg_cost_usd']:.4f}"
            else:
                cost_str = "N/A"
            
            table.add_row(
                result["experiment_id"],
                str(result["n_examples"]),
                metrics_str,
                f"{avg_latency:.1f}",
                f"{total_tokens:.0f}",
                cost_str,
            )
        
        console.print(table)
        
        # Display per-task breakdown for multi-task experiments
        for result in results:
            if result.get("is_multi_task", False):
                console.print(f"\n[bold cyan]{result['experiment_id']} - Per-Task Breakdown:[/bold cyan]")
                
                task_table = Table(show_header=True, header_style="bold blue")
                task_table.add_column("Task", style="yellow")
                task_table.add_column("N", justify="right", style="white")
                task_table.add_column("Metrics", style="green")
                task_table.add_column("Avg Latency (ms)", justify="right", style="magenta")
                
                for task_name, task_result in result["per_task"].items():
                    task_metrics_str = ", ".join([f"{k}={v:.2f}" for k, v in task_result["metrics"].items()])
                    task_table.add_row(
                        task_name,
                        str(task_result["n_examples"]),
                        task_metrics_str,
                        f"{task_result['avg_latency_ms']:.1f}",
                    )
                
                console.print(task_table)
    
    if failed_experiments:
        console.print("\n[bold red]Failed Experiments:[/bold red]")
        for exp_id, error in failed_experiments:
            console.print(f"  • {exp_id}: {error}")
    
    if results:
        console.print(f"\n[bold green]✓ {len(results)} experiment(s) completed successfully[/bold green]")
        console.print(f"[dim]Results saved to results/raw/ and results/summaries/[/dim]")


if __name__ == "__main__":
    main()
