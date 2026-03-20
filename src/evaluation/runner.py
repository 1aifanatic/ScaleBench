"""Experiment runner: orchestrates dataset/task/model/metrics."""
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from core.config import ExperimentConfig, ParallelConfig
from core.registry import get_model_client, get_dataset, get_task, get_metrics
from utils.parallel import process_examples_parallel
from models.base import GenerationInput
from dataset.longbench import VALID_TASKS as LONGBENCH_TASKS
from metrics.retrieval import (
    compute_retrieval_precision,
    compute_retrieval_recall,
    compute_retrieval_f1,
    compute_combined_score,
)
import utils.mlflow_utils as mlflow_utils

# Module-level parallel config (set by run_benchmarks.py)
_parallel_config: Optional[ParallelConfig] = None


def set_parallel_config(config: Optional[ParallelConfig]):
    """Set the parallel processing configuration for this module.
    
    Args:
        config: ParallelConfig instance or None to disable parallel processing
    """
    global _parallel_config
    _parallel_config = config


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
        
        # Flush traces before run ends to ensure all async traces are persisted
        mlflow_utils.flush_traces(block=True)
        
        return result


def _process_single_example(
    example: Dict[str, Any],
    config: ExperimentConfig,
    model_client,
    task,
    metric_fns: Dict[str, Any],
) -> Dict[str, Any]:
    """Process a single example: generate, postprocess, compute metrics.
    
    This function is designed to be thread-safe and can be called concurrently.
    
    Args:
        example: Example dictionary with id, context, question, answer
        config: ExperimentConfig for the experiment
        model_client: Model client instance
        task: Task instance
        metric_fns: Dictionary of metric functions
    
    Returns:
        Result dictionary with predictions, metrics, and metadata
    """
    example_id = example["id"]
    gold_answers = example["answer"]
    
    # Wrap entire example processing in a span with gold answer
    with mlflow_utils.start_span(name=f"example_{example_id}", span_type="CHAIN") as example_span:
        # Set gold/reference answer as span attribute for visibility in traces
        example_span.set_inputs({
            "example_id": example_id,
            "gold_answer": gold_answers,
        })
        
        # Generate response
        start_time = time.time()
        
        # Build generation kwargs from experiment config.model.generation if present
        generation_kwargs = {}
        gen_cfg = getattr(config.model, "generation", None) or {}
        if "temperature" in gen_cfg and gen_cfg["temperature"] is not None:
            generation_kwargs["temperature"] = gen_cfg["temperature"]
        if "top_p" in gen_cfg and gen_cfg["top_p"] is not None:
            generation_kwargs["top_p"] = gen_cfg["top_p"]
        max_tokens_val = gen_cfg.get("max_tokens") if "max_tokens" in gen_cfg else gen_cfg.get("max_output_tokens")
        if max_tokens_val is not None:
            generation_kwargs["max_output_tokens"] = max_tokens_val
        
        # Build input via task
        generation_input = task.build_prompt(example, **generation_kwargs)
        
        output = model_client.generate(generation_input)
        latency_ms = (time.time() - start_time) * 1000
        
        # Postprocess prediction
        prediction = task.postprocess(output.text)
        
        # Check if this is a retrieval task
        is_retrieval_task = config.task.name.lower() == "retrieval_task"
        
        # Compute metrics per-example
        example_metrics = {}
        
        if is_retrieval_task and isinstance(prediction, dict):
            # Handle retrieval task
            pred_answer = prediction.get("answer") or ""
            pred_passages = prediction.get("passages")
            gold_passages = example.get("selected_indices", [])
            
            well_formed = example.get("well_formed_answers", [])
            reference_sets = [gold_answers]
            if well_formed:
                reference_sets.append(well_formed)
            
            # Compute answer metrics
            answer_metric_names = [m for m in metric_fns.keys() 
                                  if not m.startswith("retrieval_")]
            for metric_name in answer_metric_names:
                metric_fn = metric_fns[metric_name]
                best_score = {}
                for ref_set in reference_sets:
                    score_dict = metric_fn(
                        predictions=[pred_answer],
                        references=[ref_set],
                        normalize_fn=task.normalize,
                    )
                    for key, value in score_dict.items():
                        if key not in best_score or value > best_score[key]:
                            best_score[key] = value
                if metric_name in best_score:
                    example_metrics[metric_name] = best_score[metric_name]
                else:
                    for key, value in best_score.items():
                        example_metrics[key] = value
            
            # Compute retrieval metrics
            if pred_passages is not None:
                retrieval_scores = {}
                retrieval_scores.update(compute_retrieval_precision(
                    [pred_passages], [gold_passages]
                ))
                retrieval_scores.update(compute_retrieval_recall(
                    [pred_passages], [gold_passages]
                ))
                retrieval_scores.update(compute_retrieval_f1(
                    [pred_passages], [gold_passages]
                ))
                example_metrics.update(retrieval_scores)
            else:
                example_metrics["retrieval_precision"] = 0.0
                example_metrics["retrieval_recall"] = 0.0
                example_metrics["retrieval_f1"] = 0.0
            
            # Compute combined score
            answer_f1 = example_metrics.get("f1", 0.0)
            retrieval_f1 = example_metrics.get("retrieval_f1", 0.0)
            weights = config.combined_score_weights or {}
            combined = compute_combined_score(
                answer_f1=answer_f1,
                retrieval_f1=retrieval_f1,
                answer_weight=weights.get("answer_weight", 0.7),
                retrieval_weight=weights.get("retrieval_weight", 0.3),
            )
            example_metrics.update(combined)
        else:
            # Standard task
            for metric_name, metric_fn in metric_fns.items():
                score_dict = metric_fn(
                    predictions=[prediction],
                    references=[gold_answers],
                    normalize_fn=task.normalize,
                )
                if metric_name in score_dict:
                    example_metrics[metric_name] = score_dict[metric_name]
                else:
                    for score_key, score_value in score_dict.items():
                        example_metrics[score_key] = score_value
        
        # Set span outputs
        example_span.set_outputs({
            "prediction": prediction,
            "metrics": example_metrics,
            "latency_ms": latency_ms,
        })
    
    # Build result dictionary
    if is_retrieval_task and isinstance(prediction, dict):
        result = {
            "id": example_id,
            "model": config.model.name,
            "dataset": config.dataset.name,
            "task": config.task.name,
            "mode": config.model.mode,
            "question": example.get("question", ""),
            "pred_answer": prediction.get("answer"),
            "pred_passages": prediction.get("passages"),
            "pred": prediction,
            "gold": gold_answers,
            "gold_well_formed_answers": example.get("well_formed_answers", []),
            "gold_passages": example.get("selected_indices", []),
            "input_tokens": output.input_tokens,
            "output_tokens": output.output_tokens,
            "latency_ms": latency_ms,
            "cost_usd": output.cost_usd,
            "metrics": example_metrics,
        }
    else:
        result = {
            "id": example_id,
            "model": config.model.name,
            "dataset": config.dataset.name,
            "task": config.task.name,
            "mode": config.model.mode,
            "question": example.get("question", ""),
            "pred": prediction,
            "gold": gold_answers,
            "input_tokens": output.input_tokens,
            "output_tokens": output.output_tokens,
            "latency_ms": latency_ms,
            "cost_usd": output.cost_usd,
            "metrics": example_metrics,
        }
        if "context_source" in example:
            result["context_source"] = example["context_source"]
    
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
    
    # Load examples and convert to list (generator can't be iterated multiple times)
    examples = list(dataset.load_examples(
        split=config.dataset.split,
        limit=config.dataset.num_examples,
        seed=config.dataset.seed,
    ))
    
    # Open output file for writing (resume support: skip existing IDs)
    existing_ids = set()
    if raw_output_file.exists():
        with open(raw_output_file, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    existing_ids.add(entry["id"])
    
    # Determine if we should use parallel processing
    use_parallel = _parallel_config is not None and _parallel_config.enabled
    
    if use_parallel:
        # Parallel processing path
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Using parallel processing with {_parallel_config.max_workers} workers")
        
        # Create process function with closure over config and components
        def process_example(example):
            return _process_single_example(
                example=example,
                config=config,
                model_client=model_client,
                task=task,
                metric_fns=metric_fns,
            )
        
        # Process examples in parallel
        results = process_examples_parallel(
            examples=examples,
            process_fn=process_example,
            output_file=raw_output_file,
            existing_ids=existing_ids,
            max_workers=_parallel_config.max_workers,
            retry_attempts=_parallel_config.retry_attempts,
            retry_base_delay=_parallel_config.retry_base_delay,
            retry_max_delay=_parallel_config.retry_max_delay,
        )
    else:
        # Sequential processing path (original implementation)
        # Append mode for resume support
        with open(raw_output_file, "a") as f:
            for example in examples:
                example_id = example["id"]
                
                # Skip if already processed
                if example_id in existing_ids:
                    continue
                
                # Process example using extracted function
                result = _process_single_example(
                    example=example,
                    config=config,
                    model_client=model_client,
                    task=task,
                    metric_fns=metric_fns,
                )
                
                results.append(result)
                
                # Write to JSONL
                f.write(json.dumps(result) + "\n")
                f.flush()  # Ensure immediate write for checkpointing
    
    # Aggregate results
    aggregated = _aggregate_results(results, config)
    aggregated["raw_output_file"] = str(raw_output_file)
    aggregated["results"] = results

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
