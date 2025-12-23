"""Schema definitions and conversion utilities for leaderboard data.

This module handles conversion of benchmark summary JSONs to leaderboard-friendly
flat row format. Designed to handle variable metrics dynamically.
"""
from typing import Dict, Any, Optional
from datetime import datetime


def extract_experiment_metadata(experiment_id: str) -> Dict[str, str]:
    """Extract metadata from experiment ID.
    
    Experiment IDs follow patterns:
    - Standard: {dataset}_{mode}_{model}_{timestamp}
      Example: hotpotqa_baseline_openai_20251215_130236
    - LongBench: {dataset}_{task}_{mode}_{model}_{timestamp}
      Example: longbench_narrativeqa_baseline_openai_20251223_134109
    
    Args:
        experiment_id: Full experiment identifier.
        
    Returns:
        Dictionary with dataset, mode, task, model_short, and timestamp.
    """
    parts = experiment_id.split("_")
    
    if len(parts) < 4:
        return {
            "dataset": "unknown",
            "mode": "unknown",
            "task": "",
            "model_short": "unknown",
            "timestamp": ""
        }
    
    # Timestamp is always last 2 parts (YYYYMMDD_HHMMSS)
    timestamp = f"{parts[-2]}_{parts[-1]}"
    dataset = parts[0]
    
    # Detect LongBench pattern: longbench_{task}_{mode}_{model}_{timestamp}
    if dataset == "longbench" and len(parts) >= 6:
        task = parts[1]
        mode = parts[2]
        model_short = "_".join(parts[3:-2])
    else:
        # Standard pattern: {dataset}_{mode}_{model}_{timestamp}
        task = ""
        mode = parts[1]
        model_short = "_".join(parts[2:-2])
    
    return {
        "dataset": dataset,
        "mode": mode,
        "task": task,
        "model_short": model_short,
        "timestamp": timestamp
    }


def summary_to_leaderboard_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Convert summary JSON to flat leaderboard row.
    
    This function dynamically handles variable metrics by:
    1. Extracting all metric means from the metrics dict
    2. Preserving standard fields (model, dataset, task, cost, latency, tokens)
    3. Supporting optional fields (compression_ratio, etc.)
    4. Handling missing fields gracefully for backwards compatibility
    
    Args:
        summary: Summary dictionary from aggregation.py output.
        
    Returns:
        Flat dictionary suitable for leaderboard display.
    """
    experiment_id = summary.get("experiment_id", "")
    metadata = extract_experiment_metadata(experiment_id)
    
    # Base row with standard fields
    # Prefer fields from summary, fall back to parsed metadata
    row = {
        "experiment_id": experiment_id,
        "model": summary.get("model", ""),
        "dataset": summary.get("dataset", metadata["dataset"]),
        "mode": summary.get("mode", metadata["mode"]),  # Use summary mode first
        "task": summary.get("task", metadata["task"]),  # Use summary task first
        "n_examples": summary.get("n_examples", 0),
        "timestamp": metadata["timestamp"],
    }
    
    # Extract all metric means dynamically
    metrics = summary.get("metrics", {})
    for metric_name, metric_data in metrics.items():
        if isinstance(metric_data, dict):
            # Standard format with mean/std/etc
            row[f"{metric_name}_mean"] = metric_data.get("mean", 0.0)
            row[f"{metric_name}_std"] = metric_data.get("std", 0.0)
        else:
            # Direct value (for backwards compatibility)
            row[f"{metric_name}_mean"] = metric_data
    
    # Add latency stats
    latency_ms = summary.get("latency_ms", {})
    if isinstance(latency_ms, dict):
        row["avg_latency_ms"] = latency_ms.get("mean", 0.0)
        row["latency_p50"] = latency_ms.get("p50", 0.0)
        row["latency_p95"] = latency_ms.get("p95", 0.0)
    else:
        row["avg_latency_ms"] = latency_ms
    
    # Add token stats
    tokens = summary.get("tokens", {})
    row["avg_input_tokens"] = tokens.get("avg_input_tokens", 0.0)
    row["avg_output_tokens"] = tokens.get("avg_output_tokens", 0.0)
    
    # Add cost stats
    cost = summary.get("cost", {})
    if cost:
        row["total_cost_usd"] = cost.get("total_usd", 0.0)
        row["avg_cost_usd"] = cost.get("mean_usd", 0.0)
    else:
        row["total_cost_usd"] = 0.0
        row["avg_cost_usd"] = 0.0
    
    # Add compression ratio from summary
    # For backwards compatibility: if not present, infer from mode
    if "compression_ratio" in summary:
        row["compression_ratio"] = summary["compression_ratio"]
    else:
        # Old results without compression_ratio field
        mode = row["mode"]
        if mode == "baseline":
            row["compression_ratio"] = "N/A"
        elif mode == "scaledown":
            row["compression_ratio"] = "auto"  # Default assumption
        else:
            row["compression_ratio"] = "N/A"
    
    # Add upload timestamp
    row["uploaded_at"] = datetime.now().isoformat()
    
    return row


def leaderboard_row_to_json(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert leaderboard row back to JSON for storage.
    
    This is for saving to HF dataset. Includes metadata for reconstruction.
    
    Args:
        row: Flattened leaderboard row.
        
    Returns:
        JSON-serializable dictionary.
    """
    return row
