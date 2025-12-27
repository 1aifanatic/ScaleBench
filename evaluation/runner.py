"""Experiment runner: orchestrates dataset/task/model/metrics."""
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from core.config import ExperimentConfig
from core.registry import get_model_client, get_dataset, get_task, get_metrics
from models.base import GenerationInput
from dataset.longbench import VALID_TASKS as LONGBENCH_TASKS
import utils.mlflow_utils as mlflow_utils


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run a single experiment and return aggregated results.
    
    Handles both single-task and multi-task (LongBench) experiments.
    For LongBench with task=None, runs all 4 QA tasks and aggregates.
    
    Args:
        config: ExperimentConfig with model/dataset/task/metrics configuration.
        
    Returns:
        Dictionary with aggregated results including per-example raw data.
    """
    # Check if this is a multi-task LongBench experiment
    is_longbench_multi = (
        config.dataset.name == "longbench" and 
        config.dataset.task is None
    )
    
    # Prepare MLflow run tags
    tags = {
        "dataset": config.dataset.name,
        "mode": config.model.mode,
        "model_provider": config.model.name.split(":")[0] if ":" in config.model.name else "unknown",
    }
    
    # Wrap experiment execution with MLflow run
    with mlflow_utils.start_run(run_name=config.id, tags=tags):
        # Log experiment parameters
        params = {
            "experiment_id": config.id,
            "dataset": config.dataset.name,
            "dataset_split": config.dataset.split,
            "num_examples": str(config.dataset.num_examples or "all"),
            "seed": config.dataset.seed,
            "model": config.model.name,
            "mode": config.model.mode,
            "task": config.task.name,
            "metrics": ",".join(config.metrics) if config.metrics else "",
        }
        
        # Add dataset task if LongBench
        if config.dataset.task:
            params["dataset_task"] = config.dataset.task
        
        # Add ScaleDown params if applicable
        if config.model.mode == "scaledown" and config.model.scaledown:
            params["compression_rate"] = str(config.model.scaledown.get("rate", "auto"))
            params["compression_model"] = config.model.scaledown.get("compression_model", "gemini-2.5-flash")
        
        mlflow_utils.log_params(params)
        
        if is_longbench_multi:
            result = _run_multi_task_experiment(config)
        else:
            result = _run_single_task_experiment(config)
        
        # Log aggregated metrics to MLflow
        _log_mlflow_metrics(result, config)
        
        # Log artifacts
        _log_mlflow_artifacts(result)
        
        return result


def _run_single_task_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run a single-task experiment.
    
    Args:
        config: ExperimentConfig with model/dataset/task/metrics configuration.
        
    Returns:
        Dictionary with aggregated results.
    """
    # Instantiate components
    model_client = get_model_client(config.model)
    dataset = get_dataset(config.dataset)
    task = get_task(config.task, dataset_name=config.dataset.name)
    metric_fns = get_metrics(config.metrics)
    
    # Setup output paths with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_output_dir = Path("results/raw")
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_file = raw_output_dir / f"{config.id}_{timestamp}.jsonl"
    
    # Collect per-example results
    results = []
    
    # Load examples and iterate
    examples = dataset.load_examples(
        split=config.dataset.split,
        limit=config.dataset.num_examples,
        seed=config.dataset.seed,
    )
    
    # Open output file for writing (resume support: skip existing IDs)
    existing_ids = set()
    if raw_output_file.exists():
        with open(raw_output_file, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    existing_ids.add(entry["id"])
    
    # Append mode for resume support
    with open(raw_output_file, "a") as f:
        for example in examples:
            example_id = example["id"]
            
            # Skip if already processed
            if example_id in existing_ids:
                continue
            
            # Get gold answers early so we can include them in the trace
            gold_answers = example["answer"]
            
            # Wrap entire example processing in a span with gold answer
            with mlflow_utils.start_span(name=f"example_{example_id}", span_type="CHAIN") as example_span:
                # Set gold/reference answer as span attribute for visibility in traces
                example_span.set_inputs({
                    "example_id": example_id,
                    "gold_answer": gold_answers,  # Reference/expected answer
                })
                
                # Generate response
                start_time = time.time()
                
                # Build generation kwargs from experiment config.model.generation if present
                generation_kwargs = {}
                gen_cfg = getattr(config.model, "generation", None) or {}
                # Map config keys to GenerationInput fields
                if "temperature" in gen_cfg and gen_cfg["temperature"] is not None:
                    generation_kwargs["temperature"] = gen_cfg["temperature"]
                if "top_p" in gen_cfg and gen_cfg["top_p"] is not None:
                    generation_kwargs["top_p"] = gen_cfg["top_p"]
                # Support both `max_tokens` (YAML-friendly) and `max_output_tokens` as synonyms
                max_tokens_val = gen_cfg.get("max_tokens") if "max_tokens" in gen_cfg else gen_cfg.get("max_output_tokens")
                if max_tokens_val is not None:
                    generation_kwargs["max_output_tokens"] = max_tokens_val

                # Build input via task (handles prompt ordering and context separation)
                generation_input = task.build_prompt(example, **generation_kwargs)
                
                output = model_client.generate(generation_input)
                latency_ms = (time.time() - start_time) * 1000
                
                # Postprocess prediction
                prediction = task.postprocess(output.text)
                
                # Compute metrics per-example
                example_metrics = {}
                for metric_name, metric_fn in metric_fns.items():
                    # Pass predictions and references as lists
                    score_dict = metric_fn(
                        predictions=[prediction],
                        references=[gold_answers],
                        normalize_fn=task.normalize,
                    )
                    # Handle metrics that return multiple scores (e.g., bertscore)
                    if metric_name in score_dict:
                        # Single score with matching key (e.g., f1 -> {"f1": 0.95})
                        example_metrics[metric_name] = score_dict[metric_name]
                    else:
                        # Multiple scores (e.g., bertscore -> {"bertscore_precision": ..., "bertscore_recall": ..., "bertscore_f1": ...})
                        for score_key, score_value in score_dict.items():
                            example_metrics[score_key] = score_value
                
                # Set span outputs with prediction and metrics
                example_span.set_outputs({
                    "prediction": prediction,
                    "metrics": example_metrics,
                    "latency_ms": latency_ms,
                })
            
            # Collect result (outside span, just data collection)
            result = {
                "id": example_id,
                "model": config.model.name,
                "dataset": config.dataset.name,
                "task": config.task.name,
                "mode": config.model.mode,
                "input_system_prompt": output.input_system_prompt,
                "input_question": output.input_question,
                "input_context": output.input_context,
                "pred": prediction,
                "gold": gold_answers,
                "input_tokens": output.input_tokens,
                "output_tokens": output.output_tokens,
                "latency_ms": latency_ms,
                "cost_usd": output.cost_usd,
                "metrics": example_metrics,
            }
            
            results.append(result)
            
            # Write to JSONL
            f.write(json.dumps(result) + "\n")
            f.flush()  # Ensure immediate write for checkpointing
    
    # Aggregate results
    aggregated = _aggregate_results(results, config)
    aggregated["raw_output_file"] = str(raw_output_file)
    
    return aggregated


def _aggregate_results(results: List[Dict[str, Any]], config: ExperimentConfig) -> Dict[str, Any]:
    """Aggregate per-example results into summary statistics.
    
    Args:
        results: List of per-example result dictionaries.
        config: ExperimentConfig for metadata.
        
    Returns:
        Dictionary with aggregated metrics and metadata.
    """
    if not results:
        return {
            "experiment_id": config.id,
            "model": config.model.name,
            "dataset": config.dataset.name,
            "task": config.task.name,
            "mode": config.model.mode,
            "n_examples": 0,
            "metrics": {},
            "latency_ms": 0.0,
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "compression_ratio": _get_compression_ratio(config),
        }
    
    n = len(results)
    
    # Aggregate metrics
    metric_sums = {}
    for result in results:
        for metric_name, score in result["metrics"].items():
            metric_sums[metric_name] = metric_sums.get(metric_name, 0.0) + score
    
    metric_means = {name: total / n for name, total in metric_sums.items()}
    
    # Aggregate latency and tokens
    total_latency = sum(r["latency_ms"] for r in results)
    total_input_tokens = sum(r["input_tokens"] for r in results)
    total_output_tokens = sum(r["output_tokens"] for r in results)
    
    # Aggregate cost (filter out None values)
    costs = [r.get("cost_usd") for r in results if r.get("cost_usd") is not None]
    total_cost = sum(costs) if costs else None
    avg_cost = total_cost / n if costs else None
    
    return {
        "experiment_id": config.id,
        "model": config.model.name,
        "dataset": config.dataset.name,
        "task": config.task.name,
        "mode": config.model.mode,
        "n_examples": n,
        "metrics": metric_means,
        "avg_latency_ms": total_latency / n,
        "avg_input_tokens": total_input_tokens / n,
        "avg_output_tokens": total_output_tokens / n,
        "total_cost_usd": total_cost,
        "avg_cost_usd": avg_cost,
        "compression_ratio": _get_compression_ratio(config),
    }


def _get_compression_ratio(config: ExperimentConfig) -> Optional[str]:
    """Get compression ratio from config.
    
    Args:
        config: ExperimentConfig
        
    Returns:
        Compression ratio: N/A for baseline, "auto" or integer (1-10) for scaledown
    """
    if config.model.mode == "baseline":
        return "N/A"
    elif config.model.mode == "scaledown" and config.model.scaledown:
        rate = config.model.scaledown.get("rate", "auto")
        if rate == "auto":
            return "auto"
        else:
            # Convert to integer if it's a numeric rate
            try:
                return str(int(float(rate)))
            except (ValueError, TypeError):
                return "auto"
    return "N/A"


def _run_multi_task_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run multi-task LongBench experiment (all 4 QA tasks).
    
    Args:
        config: ExperimentConfig with LongBench dataset and task=None.
        
    Returns:
        Dictionary with per-task and overall aggregated results.
    """
    from dataclasses import replace
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    per_task_results = {}
    all_raw_files = []
    
    # Run each task separately
    for task_name in LONGBENCH_TASKS:
        # Create modified config with specific task
        task_config = replace(
            config,
            dataset=replace(config.dataset, task=task_name)
        )
        
        # Run single-task experiment
        task_result = _run_single_task_experiment(task_config)
        
        # Store per-task results
        per_task_results[task_name] = task_result
        all_raw_files.append(task_result["raw_output_file"])
    
    # Aggregate across all tasks
    overall_metrics = {}
    overall_n_examples = 0
    overall_latency = 0.0
    overall_input_tokens = 0.0
    overall_output_tokens = 0.0
    overall_total_cost = 0.0
    
    for task_name, task_result in per_task_results.items():
        n = task_result["n_examples"]
        overall_n_examples += n
        
        # Accumulate metrics (weighted by n_examples)
        for metric_name, value in task_result["metrics"].items():
            if metric_name not in overall_metrics:
                overall_metrics[metric_name] = 0.0
            overall_metrics[metric_name] += value * n
        
        # Accumulate other stats
        overall_latency += task_result["avg_latency_ms"] * n
        overall_input_tokens += task_result["avg_input_tokens"] * n
        overall_output_tokens += task_result["avg_output_tokens"] * n
        
        if task_result.get("total_cost_usd") is not None:
            overall_total_cost += task_result["total_cost_usd"]
    
    # Compute overall averages
    if overall_n_examples > 0:
        overall_metrics = {name: total / overall_n_examples for name, total in overall_metrics.items()}
        overall_avg_latency = overall_latency / overall_n_examples
        overall_avg_input_tokens = overall_input_tokens / overall_n_examples
        overall_avg_output_tokens = overall_output_tokens / overall_n_examples
        overall_avg_cost = overall_total_cost / overall_n_examples if overall_total_cost > 0 else None
    else:
        overall_avg_latency = 0.0
        overall_avg_input_tokens = 0.0
        overall_avg_output_tokens = 0.0
        overall_avg_cost = None
    
    return {
        "experiment_id": config.id,
        "model": config.model.name,
        "dataset": config.dataset.name,
        "is_multi_task": True,
        "n_examples": overall_n_examples,
        "metrics": overall_metrics,
        "avg_latency_ms": overall_avg_latency,
        "avg_input_tokens": overall_avg_input_tokens,
        "avg_output_tokens": overall_avg_output_tokens,
        "total_cost_usd": overall_total_cost if overall_total_cost > 0 else None,
        "avg_cost_usd": overall_avg_cost,
        "per_task": per_task_results,
        "raw_output_files": all_raw_files,
    }


def _log_mlflow_metrics(result: Dict[str, Any], config: ExperimentConfig) -> None:
    """Log aggregated metrics to MLflow.
    
    Args:
        result: Aggregated results dictionary.
        config: ExperimentConfig for metadata.
    """
    if not mlflow_utils.is_mlflow_enabled():
        return
    
    metrics = {}
    
    # Log metric means
    for metric_name, value in result.get("metrics", {}).items():
        if isinstance(value, dict):
            # New format with mean/std
            metrics[f"{metric_name}_mean"] = value.get("mean", 0.0)
        else:
            # Old format (direct value)
            metrics[metric_name] = value
    
    # Log latency
    if "latency_ms" in result:
        metrics["latency_mean_ms"] = result["latency_ms"].get("mean", 0.0)
    elif "avg_latency_ms" in result:
        metrics["latency_mean_ms"] = result["avg_latency_ms"]
    
    # Log tokens
    if "tokens" in result:
        metrics["avg_input_tokens"] = result["tokens"].get("avg_input_tokens", 0.0)
        metrics["avg_output_tokens"] = result["tokens"].get("avg_output_tokens", 0.0)
    else:
        metrics["avg_input_tokens"] = result.get("avg_input_tokens", 0.0)
        metrics["avg_output_tokens"] = result.get("avg_output_tokens", 0.0)
    
    # Log cost
    if result.get("avg_cost_usd") is not None:
        metrics["avg_cost_usd"] = result["avg_cost_usd"]
    if result.get("total_cost_usd") is not None:
        metrics["total_cost_usd"] = result["total_cost_usd"]
    
    # Log number of examples
    metrics["n_examples"] = result.get("n_examples", 0)
    
    mlflow_utils.log_metrics(metrics)


def _log_mlflow_artifacts(result: Dict[str, Any]) -> None:
    """Log result files as MLflow artifacts.
    
    Args:
        result: Result dictionary containing file paths.
    """
    if not mlflow_utils.is_mlflow_enabled():
        return
    
    # Log raw output file(s)
    if "raw_output_file" in result:
        mlflow_utils.log_artifact(result["raw_output_file"])
    
    if "raw_output_files" in result:
        for file_path in result["raw_output_files"]:
            mlflow_utils.log_artifact(file_path)
