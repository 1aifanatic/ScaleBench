"""CLI entrypoint: loads config, runs experiments."""
import argparse
import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Load .env file before anything else so API keys are available
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file, override=False)
    except ImportError:
        pass

# Ensure src/ is on the path when run directly
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from core.config import load_config
from evaluation.runner import run_experiment, set_parallel_config
from evaluation.aggregation import (
    aggregate_from_jsonl,
    save_summary_csv,
    save_summary_json,
    save_predictions_csv,
)
from huggingface.uploader import upload_result
import utils.mlflow_utils as mlflow_utils
import utils.langsmith_utils as langsmith_utils

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
        "--tag",
        type=str,
        default=None,
        help="Run all experiments with the given tag (e.g. --tag qmsum).",
    )
    parser.add_argument(
        "--parallel-experiments",
        action="store_true",
        help="Run the selected experiments concurrently (one thread per experiment).",
    )
    parser.add_argument(
        "--serial-experiments",
        action="store_true",
        help="Run the selected experiments one after another (default behaviour).",
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
    parser.add_argument(
        "--langsmith",
        action="store_true",
        help="Enable LangSmith tracing (overrides config file).",
    )
    parser.add_argument(
        "--no-langsmith",
        action="store_true",
        help="Disable LangSmith tracing (overrides config file).",
    )
    parser.add_argument(
        "--langsmith-project",
        type=str,
        default=None,
        help="LangSmith project name (default: LANGSMITH_PROJECT env var or 'scalebench').",
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

    # Set parallel processing config (per-example parallelism, from YAML)
    if config.parallel:
        set_parallel_config(config.parallel)
        if config.parallel.enabled:
            console.print(
                f"[green]Parallel processing enabled: "
                f"{config.parallel.max_workers} workers[/green]\n"
            )

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
            console.print(
                "[yellow]MLflow initialization failed, continuing without MLflow"
                "[/yellow]\n"
            )

    # Initialize LangSmith if enabled
    langsmith_enabled = _should_enable_langsmith(args, config)
    if langsmith_enabled:
        ls_project = (
            args.langsmith_project
            or os.getenv("LANGSMITH_PROJECT")
            or (config.langsmith and config.langsmith.project_name)
            or "scalebench"
        )
        ls_api_key = config.langsmith.api_key if config.langsmith else None
        success = langsmith_utils.setup_langsmith(
            enabled=True,
            project_name=ls_project,
            api_key=ls_api_key,
        )
        if success:
            console.print(
                f"[green]LangSmith enabled: project={ls_project}[/green]\n"
            )
        else:
            console.print(
                "[yellow]LangSmith initialization failed, continuing "
                "without LangSmith[/yellow]\n"
            )

    # Filter experiments
    experiments_to_run = config.experiments

    if args.experiment_id:
        experiments_to_run = [
            exp for exp in config.experiments if exp.id == args.experiment_id
        ]
        if not experiments_to_run:
            console.print(
                f"[bold red]Experiment '{args.experiment_id}' not found in config."
                f"[/bold red]"
            )
            console.print(f"Available: {[e.id for e in config.experiments]}")
            sys.exit(1)
        console.print(
            f"[yellow]Filtered to experiment: {args.experiment_id}[/yellow]\n"
        )

    elif args.tag:
        experiments_to_run = [
            exp for exp in config.experiments
            if args.tag in (exp.tags or [])
        ]
        if not experiments_to_run:
            console.print(
                f"[bold red]No experiments found with tag '{args.tag}'.[/bold red]"
            )
            all_tags = sorted({t for e in config.experiments for t in (e.tags or [])})
            console.print(f"Available tags: {all_tags}")
            sys.exit(1)
        console.print(
            f"[yellow]Filtered to tag '{args.tag}': "
            f"{len(experiments_to_run)} experiment(s)[/yellow]\n"
        )

    # Determine execution mode (parallel-experiments vs serial)
    run_parallel = args.parallel_experiments and not args.serial_experiments

    # Display experiment summary
    _display_experiment_summary(experiments_to_run, run_parallel=run_parallel)

    if args.dry_run:
        console.print(
            "\n[bold yellow]DRY RUN:[/bold yellow] Config validated successfully. "
            "No experiments executed."
        )
        sys.exit(0)

    # Run experiments
    results = []
    failed_experiments = []

    if run_parallel:
        results, failed_experiments = _run_experiments_parallel(
            experiments_to_run, args, config
        )
    else:
        results, failed_experiments = _run_experiments_serial(
            experiments_to_run, args, config
        )

    # Final flush of any remaining async traces before displaying results
    if mlflow_enabled:
        mlflow_utils.flush_traces(block=True)
    if langsmith_enabled:
        langsmith_utils.flush_traces()

    # Display final summary
    console.print("\n")
    _display_results_summary(results, failed_experiments)

    # Exit with error if any experiments failed
    if failed_experiments:
        sys.exit(1)


def _process_experiment_result(exp, result, args, config):
    """Aggregate, save, and optionally upload a single experiment result.

    Returns the aggregated result dict.
    """
    is_multi_task = result.get("is_multi_task", False)

    if is_multi_task:
        first_raw_file = Path(result["raw_output_files"][0])
        timestamp = first_raw_file.stem.rsplit("_", 1)[-1]

        summary_dir = Path("results/summaries")
        summary_csv_path = summary_dir / f"{exp.id}_{timestamp}_summary.csv"
        summary_json_path = summary_dir / f"{exp.id}_{timestamp}_summary.json"
        save_summary_csv(result, summary_csv_path)
        save_summary_json(result, summary_json_path)

        if _should_upload_to_hf(args, config):
            _upload_to_hf(summary_json_path, args.hf_token, exp.id)

        return result
    else:
        raw_jsonl = Path(result["raw_output_file"])
        aggregated = aggregate_from_jsonl(raw_jsonl)

        if (
            "compression_ratio" not in aggregated
            or aggregated["compression_ratio"] == "auto"
        ):
            from evaluation.runner import _get_compression_ratio
            aggregated["compression_ratio"] = _get_compression_ratio(exp)

        timestamp = raw_jsonl.stem.rsplit("_", 1)[-1]
        summary_dir = Path("results/summaries")
        summary_csv_path = summary_dir / f"{exp.id}_{timestamp}_summary.csv"
        summary_json_path = summary_dir / f"{exp.id}_{timestamp}_summary.json"
        predictions_csv_path = (
            summary_dir / f"{exp.id}_{timestamp}_predictions.csv"
        )
        save_summary_csv(aggregated, summary_csv_path)
        save_summary_json(aggregated, summary_json_path)
        save_predictions_csv(result.get("results", []), predictions_csv_path)

        if _should_upload_to_hf(args, config):
            _upload_to_hf(summary_json_path, args.hf_token, exp.id)

        return aggregated


def _run_experiments_serial(experiments_to_run, args, config):
    """Run experiments one after another, showing a spinner per experiment."""
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
                result = run_experiment(exp)
                is_multi_task = result.get("is_multi_task", False)
                label = f"[green]✓[/green] {exp.id}"
                if is_multi_task:
                    label += " (multi-task)"
                progress.update(task, description=label)
                aggregated = _process_experiment_result(exp, result, args, config)
                results.append(aggregated)
            except Exception as e:
                progress.update(task, description=f"[red]✗[/red] {exp.id}")
                failed_experiments.append((exp.id, str(e)))
                logger.error(
                    f"Experiment {exp.id} failed: {e}", exc_info=args.verbose
                )

    return results, failed_experiments


def _run_experiments_parallel(experiments_to_run, args, config):
    """Run experiments concurrently using a thread pool."""
    results = []
    failed_experiments = []

    console.print(
        f"[cyan]Running {len(experiments_to_run)} experiments in parallel...[/cyan]\n"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Pre-create progress tasks so they all show up immediately
        task_map = {
            exp.id: progress.add_task(f"Pending  {exp.id}...", total=None)
            for exp in experiments_to_run
        }

        with ThreadPoolExecutor(max_workers=len(experiments_to_run)) as executor:
            future_to_exp = {
                executor.submit(run_experiment, exp): exp
                for exp in experiments_to_run
            }

            for future in as_completed(future_to_exp):
                exp = future_to_exp[future]
                pt = task_map[exp.id]
                try:
                    result = future.result()
                    is_multi_task = result.get("is_multi_task", False)
                    label = f"[green]✓[/green] {exp.id}"
                    if is_multi_task:
                        label += " (multi-task)"
                    progress.update(pt, description=label)
                    aggregated = _process_experiment_result(
                        exp, result, args, config
                    )
                    results.append(aggregated)
                except Exception as e:
                    progress.update(pt, description=f"[red]✗[/red] {exp.id}")
                    failed_experiments.append((exp.id, str(e)))
                    logger.error(
                        f"Experiment {exp.id} failed: {e}", exc_info=args.verbose
                    )

    return results, failed_experiments


def _should_enable_mlflow(args, config) -> bool:
    """Determine if MLflow should be enabled."""
    if args.no_mlflow:
        return False
    if args.mlflow:
        return True
    if config.mlflow and config.mlflow.enabled:
        return True
    return False


def _should_enable_langsmith(args, config) -> bool:
    """Determine if LangSmith should be enabled."""
    if args.no_langsmith:
        return False
    if args.langsmith:
        return True
    if config.langsmith and config.langsmith.enabled:
        return True
    return False


def _should_upload_to_hf(args, config) -> bool:
    """Determine if results should be uploaded to HF."""
    if args.no_upload_hf:
        return False
    if args.upload_hf:
        return True
    return getattr(config, "upload_to_hf", False)


def _upload_to_hf(summary_path: Path, token: Optional[str], experiment_id: str):
    """Upload result to HuggingFace dataset."""
    try:
        logger.info(f"Uploading {experiment_id} to HuggingFace...")
        success = upload_result(summary_path, token=token, force=False)
        if success:
            console.print("  [green]✓[/green] Uploaded to HuggingFace")
        else:
            console.print("  [yellow]○[/yellow] Already uploaded or skipped")
    except Exception as e:
        logger.error(f"Failed to upload {experiment_id} to HuggingFace: {e}")
        console.print(f"  [red]✗[/red] HF upload failed: {e}")


def _display_experiment_summary(experiments, run_parallel: bool = False):
    """Display summary table of experiments to run."""
    mode_label = "[bold magenta]PARALLEL[/bold magenta]" if run_parallel else "serial"
    table = Table(
        title=f"Experiments to Run ({mode_label})",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("ID", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Mode", style="yellow")
    table.add_column("Dataset", style="yellow")
    table.add_column("Task", style="blue")
    table.add_column("Tags", style="magenta")
    table.add_column("N", justify="right", style="white")

    for exp in experiments:
        dataset_label = exp.dataset.name
        if exp.dataset.name == "longbench":
            dataset_label += f" / {exp.dataset.task or 'all'}"

        table.add_row(
            exp.id,
            exp.model.name,
            exp.model.mode,
            f"{dataset_label} ({exp.dataset.split})",
            exp.task.name,
            ", ".join(exp.tags or []),
            str(exp.dataset.num_examples or "all"),
        )

    console.print(table)
    console.print()


def _display_results_summary(results, failed_experiments):
    """Display final results summary table."""
    if results:
        table = Table(
            title="Results Summary",
            show_header=True,
            header_style="bold green",
            expand=True,
        )
        table.add_column("Experiment", style="cyan", min_width=25)
        table.add_column("N", justify="right", style="white", width=5)
        table.add_column("Metrics", style="yellow", no_wrap=True)
        table.add_column("Avg Latency (ms)", justify="right", style="magenta", width=18)
        table.add_column("Avg Tokens", justify="right", style="blue", width=15)
        table.add_column("Avg Cost ($)", justify="right", style="green", width=15)
        table.add_column("Total Cost ($)", justify="right", style="green", width=15)

        for result in results:
            metrics_str = ", ".join([
                f"{k}={v['mean']:.3f}" if isinstance(v, dict) else f"{k}={v:.3f}"
                for k, v in result["metrics"].items()
            ])

            if "tokens" in result:
                total_tokens = (
                    result["tokens"]["avg_input_tokens"]
                    + result["tokens"]["avg_output_tokens"]
                )
            else:
                total_tokens = (
                    result.get("avg_input_tokens", 0)
                    + result.get("avg_output_tokens", 0)
                )

            if "latency_ms" in result:
                avg_latency = result["latency_ms"]["mean"]
            else:
                avg_latency = result.get("avg_latency_ms", 0)

            if "cost" in result and result["cost"]:
                avg_cost_str = f"${result['cost']['mean_usd']:.4f}"
                total_cost_str = f"${result['cost']['total_usd']:.4f}"
            elif result.get("avg_cost_usd") is not None:
                avg_cost_str = f"${result['avg_cost_usd']:.4f}"
                total_cost_str = (
                    f"${result['total_cost_usd']:.4f}"
                    if result.get("total_cost_usd") is not None
                    else "N/A"
                )
            else:
                avg_cost_str = "N/A"
                total_cost_str = "N/A"

            table.add_row(
                result["experiment_id"],
                str(result["n_examples"]),
                metrics_str,
                f"{avg_latency:.1f}",
                f"{total_tokens:.0f}",
                avg_cost_str,
                total_cost_str,
            )

        console.print(table)

        # Per-task breakdown for multi-task experiments
        for result in results:
            if result.get("is_multi_task", False):
                console.print(
                    f"\n[bold cyan]{result['experiment_id']} - "
                    f"Per-Task Breakdown:[/bold cyan]"
                )
                task_table = Table(
                    show_header=True, header_style="bold blue", expand=True
                )
                task_table.add_column("Task", style="yellow", min_width=20)
                task_table.add_column("N", justify="right", style="white", width=5)
                task_table.add_column("Metrics", style="green", no_wrap=True)
                task_table.add_column(
                    "Avg Latency (ms)", justify="right", style="magenta", width=18
                )

                for task_name, task_result in result["per_task"].items():
                    task_metrics_str = ", ".join([
                        f"{k}={v:.3f}" for k, v in task_result["metrics"].items()
                    ])
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
        console.print(
            f"\n[bold green]✓ {len(results)} experiment(s) completed successfully"
            f"[/bold green]"
        )
        console.print(
            "[dim]Results saved to results/raw/ and results/summaries/[/dim]"
        )


if __name__ == "__main__":
    main()
