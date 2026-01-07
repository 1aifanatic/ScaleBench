"""CLI entrypoint: loads config, runs experiments."""
import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from core.config import load_config
from evaluation.runner import run_experiment
from evaluation.aggregation import aggregate_from_jsonl, save_summary_csv, save_summary_json
from huggingface.uploader import upload_result
import utils.mlflow_utils as mlflow_utils

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
    parser.add_argument(
        "--upload-hf",
        action="store_true",
        help="Upload results to HuggingFace dataset after experiment completes.",
    )
    parser.add_argument(
        "--no-upload-hf",
        action="store_true",
        help="Disable HF upload even if enabled in config.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for uploads (optional, falls back to HF_TOKEN env var).",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow tracing and logging (overrides config file).",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracing (overrides config file).",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: http://localhost:5000).",
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

    # Initialize MLflow if enabled
    mlflow_enabled = _should_enable_mlflow(args, config)
    if mlflow_enabled:
        mlflow_uri = args.mlflow_uri
        if mlflow_uri is None and config.mlflow:
            mlflow_uri = config.mlflow.tracking_uri
        mlflow_uri = mlflow_uri or "http://127.0.0.1:5000"
        
        experiment_name = "ScaleDown"
        if config.mlflow and config.mlflow.experiment_name:
            experiment_name = config.mlflow.experiment_name
        
        success = mlflow_utils.setup_mlflow(
            enabled=True,
            tracking_uri=mlflow_uri,
            experiment_name=experiment_name,
        )
        if success:
            console.print(f"[green]MLflow enabled: {mlflow_uri}[/green]\n")
        else:
            console.print("[yellow]MLflow initialization failed, continuing without MLflow[/yellow]\n")

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
                    summary_csv_path = summary_dir / f"{exp.id}_{timestamp}_summary.csv"
                    summary_json_path = summary_dir / f"{exp.id}_{timestamp}_summary.json"
                    save_summary_csv(result, summary_csv_path)
                    save_summary_json(result, summary_json_path)
                    
                    # Upload to HF if requested
                    if _should_upload_to_hf(args, config):
                        _upload_to_hf(summary_json_path, args.hf_token, exp.id)
                else:
                    # Single-task: aggregate from JSONL (now includes task/mode/compression_ratio)
                    progress.update(task, description=f"[green]✓[/green] {exp.id}")
                    
                    raw_jsonl = Path(result["raw_output_file"])
                    aggregated = aggregate_from_jsonl(raw_jsonl)
                    
                    # Add compression_ratio from config if not already present
                    if "compression_ratio" not in aggregated or aggregated["compression_ratio"] == "auto":
                        from evaluation.runner import _get_compression_ratio
                        aggregated["compression_ratio"] = _get_compression_ratio(exp)
                    
                    results.append(aggregated)
                    
                    # Extract timestamp from filename (format: {exp_id}_{timestamp}.jsonl)
                    timestamp = raw_jsonl.stem.rsplit("_", 1)[-1]
                    
                    summary_dir = Path("results/summaries")
                    summary_csv_path = summary_dir / f"{exp.id}_{timestamp}_summary.csv"
                    summary_json_path = summary_dir / f"{exp.id}_{timestamp}_summary.json"
                    save_summary_csv(aggregated, summary_csv_path)
                    save_summary_json(aggregated, summary_json_path)
                    
                    # Upload to HF if requested
                    if _should_upload_to_hf(args, config):
                        _upload_to_hf(summary_json_path, args.hf_token, exp.id)
                
            except Exception as e:
                progress.update(task, description=f"[red]✗[/red] {exp.id}")
                failed_experiments.append((exp.id, str(e)))
                logger.error(f"Experiment {exp.id} failed: {e}", exc_info=args.verbose)

    # Final flush of any remaining async traces before displaying results
    if mlflow_enabled:
        mlflow_utils.flush_traces(block=True)

    # Display final summary
    console.print("\n")
    _display_results_summary(results, failed_experiments)

    # Exit with error if any experiments failed
    if failed_experiments:
        sys.exit(1)


def _should_enable_mlflow(args, config) -> bool:
    """Determine if MLflow should be enabled.
    
    Priority:
    1. CLI --no-mlflow (disable)
    2. CLI --mlflow (enable)
    3. Config mlflow.enabled setting
    4. Default: False
    """
    if args.no_mlflow:
        return False
    
    if args.mlflow:
        return True
    
    # Check config setting
    if config.mlflow and config.mlflow.enabled:
        return True
    
    return False


def _should_upload_to_hf(args, config) -> bool:
    """Determine if results should be uploaded to HF.
    
    Priority:
    1. CLI --no-upload-hf (disable)
    2. CLI --upload-hf (enable)
    3. Config upload_to_hf setting
    4. Default: False
    """
    if args.no_upload_hf:
        return False
    
    if args.upload_hf:
        return True
    
    # Check config setting
    upload_config = getattr(config, 'upload_to_hf', False)
    return upload_config


def _upload_to_hf(summary_path: Path, token: Optional[str], experiment_id: str):
    """Upload result to HuggingFace dataset."""
    try:
        logger.info(f"Uploading {experiment_id} to HuggingFace...")
        success = upload_result(
            summary_path,
            token=token,
            force=False  # Skip if already uploaded
        )
        
        if success:
            console.print(f"  [green]✓[/green] Uploaded to HuggingFace")
        else:
            console.print(f"  [yellow]○[/yellow] Already uploaded or skipped")
            
    except Exception as e:
        logger.error(f"Failed to upload {experiment_id} to HuggingFace: {e}")
        console.print(f"  [red]✗[/red] HF upload failed: {e}")


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
        table = Table(title="Results Summary", show_header=True, header_style="bold green", expand=True)
        table.add_column("Experiment", style="cyan", min_width=25)
        table.add_column("N", justify="right", style="white", width=5)
        table.add_column("Metrics", style="yellow", no_wrap=True)
        table.add_column("Avg Latency (ms)", justify="right", style="magenta", width=18)
        table.add_column("Avg Tokens", justify="right", style="blue", width=15)
        table.add_column("Avg Cost ($)", justify="right", style="green", width=15)
        
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
                
                task_table = Table(show_header=True, header_style="bold blue", expand=True)
                task_table.add_column("Task", style="yellow", min_width=20)
                task_table.add_column("N", justify="right", style="white", width=5)
                task_table.add_column("Metrics", style="green", no_wrap=True)
                task_table.add_column("Avg Latency (ms)", justify="right", style="magenta", width=18)
                
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
