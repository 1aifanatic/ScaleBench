"""Experiment runner: orchestrates dataset/task/model/metrics."""
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

from core.config import ExperimentConfig
from core.registry import get_model_client, get_dataset, get_task, get_metrics
from models.base import GenerationInput


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run a single experiment and return aggregated results.
    
    Args:
        config: ExperimentConfig with model/dataset/task/metrics configuration.
        
    Returns:
        Dictionary with aggregated results including per-example raw data.
    """
    # Instantiate components
    model_client = get_model_client(config.model)
    dataset = get_dataset(config.dataset)
    task = get_task(config.task)
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
            
            # Generate response
            start_time = time.time()
            # Pass context separately for ScaleDown compression
            # Use task to format the question with instructions
            task_prompt = task.build_prompt({"question": example.get("question", ""), "context": ""})
            
            generation_input = GenerationInput(
                user_prompt=task_prompt,
                context=example.get("context", "")
            )
            output = model_client.generate(generation_input)
            latency_ms = (time.time() - start_time) * 1000
            
            # Postprocess prediction
            prediction = task.postprocess(output.text)
            
            # Get gold answers (already list from dataset)
            gold_answers = example["answer"]
            
            # Compute metrics per-example
            example_metrics = {}
            for metric_name, metric_fn in metric_fns.items():
                # Pass predictions and references as lists
                score_dict = metric_fn(
                    predictions=[prediction],
                    references=[gold_answers],
                    normalize_fn=task.normalize,
                )
                # Extract scalar score
                example_metrics[metric_name] = score_dict[metric_name]
            
            # Collect result
            result = {
                "id": example_id,
                "model": config.model.name,
                "pred": prediction,
                "gold": gold_answers,
                "input_tokens": output.input_tokens,
                "output_tokens": output.output_tokens,
                "latency_ms": latency_ms,
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
            "n_examples": 0,
            "metrics": {},
            "latency_ms": 0.0,
            "input_tokens": 0.0,
            "output_tokens": 0.0,
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
    
    return {
        "experiment_id": config.id,
        "model": config.model.name,
        "dataset": config.dataset.name,
        "n_examples": n,
        "metrics": metric_means,
        "avg_latency_ms": total_latency / n,
        "avg_input_tokens": total_input_tokens / n,
        "avg_output_tokens": total_output_tokens / n,
    }
