"""Results aggregation and summary generation."""
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
import statistics


def aggregate_from_jsonl(jsonl_path: Path) -> Dict[str, Any]:
    """Load and aggregate results from a JSONL file.
    
    Args:
        jsonl_path: Path to raw JSONL results file.
        
    Returns:
        Dictionary with aggregated statistics.
    """
    results = []
    
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    if not results:
        return {}
    
    # Extract metadata
    experiment_id = jsonl_path.stem
    model = results[0]["model"]
    n = len(results)
    
    # Extract task, mode, compression_ratio if available (for backward compatibility)
    dataset = results[0].get("dataset", "")
    task = results[0].get("task", "")
    mode = results[0].get("mode", "")
    
    # Aggregate metrics
    metric_names = list(results[0]["metrics"].keys())
    metric_stats = {}
    
    for metric_name in metric_names:
        values = [r["metrics"][metric_name] for r in results]
        metric_stats[metric_name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "p50": statistics.median(values),
            "p95": _percentile(values, 95),
        }
    
    # Aggregate latency
    latencies = [r["latency_ms"] for r in results]
    latency_stats = {
        "mean": statistics.mean(latencies),
        "std": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "min": min(latencies),
        "max": max(latencies),
        "p50": statistics.median(latencies),
        "p95": _percentile(latencies, 95),
    }
    
    # Aggregate tokens
    input_tokens = [r["input_tokens"] for r in results]
    output_tokens = [r["output_tokens"] for r in results]
    
    token_stats = {
        "avg_input_tokens": statistics.mean(input_tokens),
        "avg_output_tokens": statistics.mean(output_tokens),
    }
    
    # Aggregate cost (filter out None values)
    costs = [r.get("cost_usd") for r in results if r.get("cost_usd") is not None]
    cost_stats = None
    if costs:
        cost_stats = {
            "total_usd": sum(costs),
            "mean_usd": statistics.mean(costs),
            "std_usd": statistics.stdev(costs) if len(costs) > 1 else 0.0,
            "min_usd": min(costs),
            "max_usd": max(costs),
            "p50_usd": statistics.median(costs),
            "p95_usd": _percentile(costs, 95),
        }
    
    aggregated = {
        "experiment_id": experiment_id,
        "model": model,
        "dataset": dataset,
        "task": task,
        "mode": mode,
        "n_examples": n,
        "metrics": metric_stats,
        "latency_ms": latency_stats,
        "tokens": token_stats,
    }
    
    if cost_stats:
        aggregated["cost"] = cost_stats
    
    # Add compression_ratio (infer from mode if not stored in results)
    if mode == "baseline":
        aggregated["compression_ratio"] = "N/A"
    elif mode == "scaledown":
        aggregated["compression_ratio"] = "auto"  # Default for old results
    else:
        aggregated["compression_ratio"] = "N/A"
    
    return aggregated


def save_summary_csv(aggregated: Dict[str, Any], output_path: Path):
    """Save aggregated results as CSV.
    
    Handles both single-task and multi-task results.
    For multi-task, saves one row per task + overall row.
    
    Args:
        aggregated: Dictionary from aggregate_from_jsonl() or multi-task results.
        output_path: Path to output CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if multi-task
    is_multi_task = aggregated.get("is_multi_task", False)
    
    if is_multi_task:
        _save_multi_task_csv(aggregated, output_path)
    else:
        _save_single_task_csv(aggregated, output_path)


def _save_single_task_csv(aggregated: Dict[str, Any], output_path: Path):
    """Save single-task results as CSV."""
    # Flatten structure for CSV
    row = {
        "experiment_id": aggregated["experiment_id"],
        "model": aggregated["model"],
        "n_examples": aggregated["n_examples"],
    }
    
    # Add metric means
    for metric_name, stats in aggregated["metrics"].items():
        row[f"{metric_name}_mean"] = stats["mean"]
        row[f"{metric_name}_std"] = stats["std"]
        row[f"{metric_name}_p50"] = stats["p50"]
        row[f"{metric_name}_p95"] = stats["p95"]
    
    # Add latency stats
    row["avg_latency_ms"] = aggregated["latency_ms"]["mean"]
    row["latency_p50"] = aggregated["latency_ms"]["p50"]
    row["latency_p95"] = aggregated["latency_ms"]["p95"]
    
    # Add token stats
    row["avg_input_tokens"] = aggregated["tokens"]["avg_input_tokens"]
    row["avg_output_tokens"] = aggregated["tokens"]["avg_output_tokens"]
    
    # Add cost stats (if available)
    if "cost" in aggregated:
        row["total_cost_usd"] = aggregated["cost"]["total_usd"]
        row["avg_cost_usd"] = aggregated["cost"]["mean_usd"]
        row["cost_p95_usd"] = aggregated["cost"]["p95_usd"]
    else:
        row["total_cost_usd"] = None
        row["avg_cost_usd"] = None
        row["cost_p95_usd"] = None
    
    # Compression ratio (if ScaleDown mode)
    row["compression_ratio"] = None
    
    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)


def _save_multi_task_csv(aggregated: Dict[str, Any], output_path: Path):
    """Save multi-task results as CSV with per-task rows + overall row."""
    rows = []
    
    # Overall row first
    overall_row = {
        "experiment_id": aggregated["experiment_id"],
        "model": aggregated["model"],
        "task": "OVERALL",
        "n_examples": aggregated["n_examples"],
    }
    
    # Add metrics
    for metric_name, value in aggregated["metrics"].items():
        overall_row[f"{metric_name}_mean"] = value
    
    overall_row["avg_latency_ms"] = aggregated["avg_latency_ms"]
    overall_row["avg_input_tokens"] = aggregated["avg_input_tokens"]
    overall_row["avg_output_tokens"] = aggregated["avg_output_tokens"]
    overall_row["total_cost_usd"] = aggregated.get("total_cost_usd")
    overall_row["avg_cost_usd"] = aggregated.get("avg_cost_usd")
    
    rows.append(overall_row)
    
    # Per-task rows
    for task_name, task_result in aggregated["per_task"].items():
        task_row = {
            "experiment_id": aggregated["experiment_id"],
            "model": aggregated["model"],
            "task": task_name,
            "n_examples": task_result["n_examples"],
        }
        
        for metric_name, value in task_result["metrics"].items():
            task_row[f"{metric_name}_mean"] = value
        
        task_row["avg_latency_ms"] = task_result["avg_latency_ms"]
        task_row["avg_input_tokens"] = task_result["avg_input_tokens"]
        task_row["avg_output_tokens"] = task_result["avg_output_tokens"]
        task_row["total_cost_usd"] = task_result.get("total_cost_usd")
        task_row["avg_cost_usd"] = task_result.get("avg_cost_usd")
        
        rows.append(task_row)
    
    # Write CSV
    with open(output_path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def save_summary_json(aggregated: Dict[str, Any], output_path: Path):
    """Save aggregated results as JSON.
    
    Handles both single-task and multi-task results.
    Multi-task results are saved with nested per_task structure.
    
    Args:
        aggregated: Dictionary from aggregate_from_jsonl() or multi-task results.
        output_path: Path to output JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)


def _percentile(values: List[float], percentile: float) -> float:
    """Calculate percentile of values.
    
    Args:
        values: List of numeric values.
        percentile: Percentile to calculate (0-100).
        
    Returns:
        Percentile value.
    """
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    if n == 0:
        return 0.0
    
    if n == 1:
        return sorted_values[0]
    
    # Use linear interpolation
    k = (n - 1) * (percentile / 100.0)
    floor = int(k)
    ceil = floor + 1
    
    if ceil >= n:
        return sorted_values[-1]
    
    weight = k - floor
    return sorted_values[floor] * (1 - weight) + sorted_values[ceil] * weight
