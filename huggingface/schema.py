"""Schema definitions and conversion utilities for leaderboard data.

This module handles conversion of benchmark summary JSONs to leaderboard-friendly
flat row format. Designed to handle variable metrics dynamically.
"""
from typing import Dict, Any
from datetime import datetime


def extract_experiment_metadata(experiment_id: str) -> Dict[str, str]:
    """Extract metadata from experiment ID.
    
    Experiment IDs follow patterns:
    - Standard: {dataset}_{mode}_{model}_{timestamp}
      Example: hotpotqa_baseline_openai_20251215_130236
    - LongBench single-task: {dataset}_{task}_{mode}_{model}_{timestamp}
      Example: longbench_narrativeqa_baseline_openai_20251223_134109
    - LongBench multi-task: {dataset}_{category}_{mode}_{model}_{timestamp}
      Example: longbench_full_baseline_openai_20251225_154113
    
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
    
    dataset = parts[0]
    
    # For LongBench, try to detect if timestamp is present (last part should be 6 digits for HHMMSS)
    # and second-to-last should be 8 digits (YYYYMMDD)
    has_valid_timestamp = False
    if len(parts) >= 2:
        try:
            last_part = parts[-1]
            second_last = parts[-2]
            # Timestamp format check: YYYYMMDD_HHMMSS
            if len(last_part) == 6 and len(second_last) == 8 and last_part.isdigit() and second_last.isdigit():
                has_valid_timestamp = True
                timestamp = f"{second_last}_{last_part}"
            else:
                timestamp = ""
        except:
            timestamp = ""
    else:
        timestamp = ""
    
    # For longbench, detect if it's multi-task (full, all, etc) or single-task
    if dataset == "longbench" and len(parts) >= 4:
        # Try to extract mode and model
        # Pattern: longbench_[category]_mode_model_[timestamp]
        # mode should be one of: baseline, scaledown
        
        if len(parts) == 4:
            # longbench_mode_model_timestamp - shouldn't happen, but handle it
            task = ""
            mode = parts[1]
            model_short = parts[2]
        elif parts[1] in ["baseline", "scaledown"]:
            # Format: longbench_mode_model_timestamp (unusual, no task/category)
            task = ""
            mode = parts[1]
            model_short = "_".join(parts[2:-2]) if len(parts) > 3 else parts[2]
        else:
            # Format: longbench_[task/category]_mode_model[_model...]_timestamp
            # parts[1] is task/category, parts[2] is mode
            task = parts[1]
            mode = parts[2]
            model_short = "_".join(parts[3:-2]) if len(parts) > 4 else parts[3]
    else:
        # Standard pattern: {dataset}_{mode}_{model}_{timestamp}
        task = ""
        mode = parts[1] if len(parts) > 1 else "unknown"
        model_short = "_".join(parts[2:-2]) if len(parts) > 3 else (parts[2] if len(parts) > 2 else "unknown")
    
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
    
    # Extract timestamp from raw_output_files if experiment_id doesn't have it
    timestamp = metadata.get("timestamp", "")
    if not timestamp:
        raw_files = summary.get("raw_output_files", [])
        if raw_files and isinstance(raw_files, list) and raw_files[0]:
            # Extract timestamp from first raw file path
            # Format: results/raw/{experiment_id}_{timestamp}.jsonl
            first_file = raw_files[0]
            parts = first_file.split("_")
            if len(parts) >= 2:
                # Last two parts should be YYYYMMDD and HHMMSS (or similar)
                possible_ts = "_".join(parts[-2:]).replace(".jsonl", "")
                if len(possible_ts.split("_")) == 2:
                    timestamp = possible_ts
    
    # Base row with standard fields
    # Prefer fields from summary, fall back to parsed metadata
    row = {
        "experiment_id": experiment_id,
        "model": summary.get("model", ""),
        "dataset": summary.get("dataset", metadata["dataset"]),
        "mode": summary.get("mode", metadata["mode"]),  # Use summary mode first
        "task": summary.get("task", metadata["task"]),  # Use summary task first
        "n_examples": summary.get("n_examples", 0),
        "timestamp": timestamp,
    }
    
    # Extract all metric means dynamically
    metrics = summary.get("metrics", {})
    for metric_name, metric_data in metrics.items():
        if isinstance(metric_data, dict):
            # Standard format with mean/std/etc
            row[f"{metric_name}_mean"] = metric_data.get("mean", 0.0)
            row[f"{metric_name}_std"] = metric_data.get("std", 0.0)
        else:
            # Direct value 
            row[f"{metric_name}_mean"] = metric_data
    
    # Add latency stats
    # Try nested format first (single-task), then direct format (multi-task)
    latency_ms = summary.get("latency_ms", {})
    if isinstance(latency_ms, dict) and latency_ms:
        row["avg_latency_ms"] = latency_ms.get("mean", 0.0)
        row["latency_p50"] = latency_ms.get("p50", 0.0)
        row["latency_p95"] = latency_ms.get("p95", 0.0)
    else:
        # Direct format (multi-task experiments)
        row["avg_latency_ms"] = summary.get("avg_latency_ms", 0.0)
        # For p50/p95, try nested format or use 0
        if isinstance(latency_ms, dict):
            row["latency_p50"] = latency_ms.get("p50", 0.0)
            row["latency_p95"] = latency_ms.get("p95", 0.0)
        else:
            row["latency_p50"] = 0.0
            row["latency_p95"] = 0.0
    
    # Add token stats
    # Try nested format first (single-task), then direct format (multi-task)
    tokens = summary.get("tokens", {})
    if isinstance(tokens, dict) and tokens:
        row["avg_input_tokens"] = tokens.get("avg_input_tokens", 0.0)
        row["avg_output_tokens"] = tokens.get("avg_output_tokens", 0.0)
    else:
        # Direct format (multi-task experiments)
        row["avg_input_tokens"] = summary.get("avg_input_tokens", 0.0)
        row["avg_output_tokens"] = summary.get("avg_output_tokens", 0.0)
    
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
