"""Data loading utilities for leaderboard.

Loads benchmark results from HuggingFace dataset and converts to DataFrames.
Handles dynamic metrics by discovering columns at load time.
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

from src.envs import TOKEN, RESULTS_DATASET, RESULTS_CACHE_PATH


def download_results(force_download: bool = False) -> List[Dict[str, Any]]:
    """Download all result JSONs from HuggingFace dataset.
    
    Args:
        force_download: If True, bypass cache and re-download files.
    
    Returns:
        List of result dictionaries.
    """
    results = []
    
    try:
        # List all files in the results/ folder of the dataset
        files = list_repo_files(repo_id=RESULTS_DATASET, repo_type="dataset", token=TOKEN)
        result_files = [f for f in files if f.startswith("results/") and f.endswith(".json")]
        
        for file_path in result_files:
            try:
                local_path = hf_hub_download(
                    repo_id=RESULTS_DATASET,
                    filename=file_path,
                    repo_type="dataset",
                    token=TOKEN,
                    local_dir=RESULTS_CACHE_PATH,
                    force_download=force_download,  # Bypass cache when needed
                )
                with open(local_path, "r") as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
                
    except Exception as e:
        print(f"Error listing dataset files: {e}")
    
    return results


def discover_metric_columns(results: List[Dict[str, Any]]) -> Set[str]:
    """Discover all unique metric columns across results.
    
    Returns:
        Set of metric column names (e.g., 'exact_match_mean', 'f1_mean').
    """
    metric_cols = set()
    for result in results:
        for key in result.keys():
            if key.endswith("_mean") or key.endswith("_std"):
                metric_cols.add(key)
    return metric_cols


def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert results list to DataFrame.
    
    Handles dynamic metrics by including all columns found in results.
    
    Args:
        results: List of flattened result dictionaries.
        
    Returns:
        DataFrame with all results.
    """
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame.from_records(results)
    
    # Round numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    
    return df


def get_leaderboard_df(force_download: bool = False) -> Tuple[pd.DataFrame, Set[str]]:
    """Get main leaderboard DataFrame.
    
    Args:
        force_download: If True, bypass cache and re-download files.
    
    Returns:
        Tuple of (DataFrame, set of metric column names).
    """
    results = download_results(force_download=force_download)
    df = results_to_dataframe(results)
    metric_cols = discover_metric_columns(results)
    
    if not df.empty:
        # Sort by first available metric or timestamp
        if "exact_match_mean" in df.columns:
            df = df.sort_values(by="exact_match_mean", ascending=False)
        elif "f1_mean" in df.columns:
            df = df.sort_values(by="f1_mean", ascending=False)
    
    return df, metric_cols


def get_comparison_df(df: pd.DataFrame) -> pd.DataFrame:
    """Create baseline vs scaledown comparison DataFrame.
    
    Pivots data to show side-by-side comparison for same model+dataset.
    
    Args:
        df: Main leaderboard DataFrame.
        
    Returns:
        Comparison DataFrame with delta columns.
    """
    if df.empty or "mode" not in df.columns:
        return pd.DataFrame()
    
    # Group by model+dataset, pivot on mode
    baseline = df[df["mode"] == "baseline"].copy()
    scaledown = df[df["mode"] == "scaledown"].copy()
    
    if baseline.empty or scaledown.empty:
        return pd.DataFrame()
    
    # Merge on model + dataset
    comparison = pd.merge(
        baseline,
        scaledown,
        on=["model", "dataset"],
        suffixes=("_baseline", "_scaledown"),
        how="inner"
    )
    
    # Calculate deltas for metric columns
    metric_cols = [c for c in baseline.columns if c.endswith("_mean")]
    for col in metric_cols:
        baseline_col = f"{col}_baseline"
        scaledown_col = f"{col}_scaledown"
        if baseline_col in comparison.columns and scaledown_col in comparison.columns:
            comparison[f"{col}_delta"] = (
                comparison[scaledown_col] - comparison[baseline_col]
            )
    
    # Calculate cost savings
    if "avg_cost_usd_baseline" in comparison.columns and "avg_cost_usd_scaledown" in comparison.columns:
        comparison["cost_savings_pct"] = (
            (comparison["avg_cost_usd_baseline"] - comparison["avg_cost_usd_scaledown"]) 
            / comparison["avg_cost_usd_baseline"] * 100
        ).round(2)
    
    return comparison
