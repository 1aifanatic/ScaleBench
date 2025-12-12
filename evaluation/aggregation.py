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
    
    return {
        "experiment_id": experiment_id,
        "model": model,
        "n_examples": n,
        "metrics": metric_stats,
        "latency_ms": latency_stats,
        "tokens": token_stats,
    }


def save_summary_csv(aggregated: Dict[str, Any], output_path: Path):
    """Save aggregated results as CSV.
    
    Args:
        aggregated: Dictionary from aggregate_from_jsonl().
        output_path: Path to output CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
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
    
    # Compression ratio (if ScaleDown mode)
    # For now, leave as placeholder since we need baseline comparison
    row["compression_ratio"] = None
    
    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)


def save_summary_json(aggregated: Dict[str, Any], output_path: Path):
    """Save aggregated results as JSON.
    
    Args:
        aggregated: Dictionary from aggregate_from_jsonl().
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
