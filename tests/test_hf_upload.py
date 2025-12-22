"""Tests for HuggingFace upload and schema conversion."""
import json
import pytest
from pathlib import Path

from huggingface.schema import extract_experiment_metadata, summary_to_leaderboard_row


def test_extract_experiment_metadata():
    """Test parsing experiment ID into metadata components."""
    # Standard format
    exp_id = "hotpotqa_baseline_openai_20251215_130236"
    meta = extract_experiment_metadata(exp_id)
    
    assert meta["dataset"] == "hotpotqa"
    assert meta["mode"] == "baseline"
    assert meta["model_short"] == "openai"
    assert meta["timestamp"] == "20251215_130236"
    
    # ScaleDown format
    exp_id = "longbench_scaledown_gemini_20251220_145523"
    meta = extract_experiment_metadata(exp_id)
    
    assert meta["dataset"] == "longbench"
    assert meta["mode"] == "scaledown"
    assert meta["model_short"] == "gemini"
    
    # Edge case: short ID
    exp_id = "test"
    meta = extract_experiment_metadata(exp_id)
    assert meta["dataset"] == "unknown"
    assert meta["mode"] == "unknown"


def test_summary_to_leaderboard_row_basic():
    """Test conversion of summary to leaderboard row with standard metrics."""
    summary = {
        "experiment_id": "hotpotqa_baseline_openai_20251215_130236",
        "model": "openai:gpt-4o",
        "dataset": "hotpotqa",
        "n_examples": 10,
        "metrics": {
            "exact_match": {
                "mean": 0.7,
                "std": 0.48,
                "min": 0.0,
                "max": 1.0,
                "p50": 1.0,
                "p95": 1.0
            },
            "f1": {
                "mean": 0.78,
                "std": 0.42,
                "min": 0.0,
                "max": 1.0,
                "p50": 1.0,
                "p95": 1.0
            }
        },
        "latency_ms": {
            "mean": 1899.06,
            "std": 1600.39,
            "min": 778.07,
            "max": 6230.88,
            "p50": 1488.21,
            "p95": 4533.56
        },
        "tokens": {
            "avg_input_tokens": 1320.6,
            "avg_output_tokens": 2.7
        },
        "cost": {
            "total_usd": 0.033285,
            "mean_usd": 0.0033285,
            "std_usd": 0.00078,
            "min_usd": 0.002345,
            "max_usd": 0.00455,
            "p50_usd": 0.003383,
            "p95_usd": 0.004452
        }
    }
    
    row = summary_to_leaderboard_row(summary)
    
    # Check standard fields
    assert row["experiment_id"] == "hotpotqa_baseline_openai_20251215_130236"
    assert row["model"] == "openai:gpt-4o"
    assert row["dataset"] == "hotpotqa"
    assert row["mode"] == "baseline"
    assert row["n_examples"] == 10
    assert row["timestamp"] == "20251215_130236"
    
    # Check dynamic metrics
    assert row["exact_match_mean"] == 0.7
    assert row["exact_match_std"] == 0.48
    assert row["f1_mean"] == 0.78
    assert row["f1_std"] == 0.42
    
    # Check performance fields
    assert row["avg_latency_ms"] == 1899.06
    assert row["latency_p50"] == 1488.21
    assert row["latency_p95"] == 4533.56
    
    # Check token fields
    assert row["avg_input_tokens"] == 1320.6
    assert row["avg_output_tokens"] == 2.7
    
    # Check cost fields
    assert row["total_cost_usd"] == 0.033285
    assert row["avg_cost_usd"] == 0.0033285
    
    # Check compression ratio
    assert row["compression_ratio"] == 1.0  # baseline mode
    
    # Check metadata
    assert "uploaded_at" in row


def test_summary_to_leaderboard_row_dynamic_metrics():
    """Test conversion handles variable metrics dynamically."""
    summary = {
        "experiment_id": "longbench_baseline_openai_20251220_145523",
        "model": "openai:gpt-4o",
        "dataset": "longbench",
        "n_examples": 10,
        "metrics": {
            "f1": {
                "mean": 0.65,
                "std": 0.25
            },
            "bertscore_precision": {
                "mean": 0.82,
                "std": 0.15
            },
            "bertscore_recall": {
                "mean": 0.79,
                "std": 0.18
            },
            "bertscore_f1": {
                "mean": 0.80,
                "std": 0.16
            }
        },
        "latency_ms": {
            "mean": 2500.0
        },
        "tokens": {
            "avg_input_tokens": 5000.0,
            "avg_output_tokens": 50.0
        },
        "cost": {
            "total_usd": 0.15,
            "mean_usd": 0.015
        }
    }
    
    row = summary_to_leaderboard_row(summary)
    
    # Check all metrics are extracted
    assert "f1_mean" in row
    assert row["f1_mean"] == 0.65
    
    assert "bertscore_precision_mean" in row
    assert row["bertscore_precision_mean"] == 0.82
    
    assert "bertscore_recall_mean" in row
    assert row["bertscore_recall_mean"] == 0.79
    
    assert "bertscore_f1_mean" in row
    assert row["bertscore_f1_mean"] == 0.80
    
    # Check std values
    assert row["f1_std"] == 0.25
    assert row["bertscore_precision_std"] == 0.15


def test_summary_to_leaderboard_row_no_cost():
    """Test conversion handles missing cost gracefully."""
    summary = {
        "experiment_id": "test_baseline_model_20251220_000000",
        "model": "test:model",
        "dataset": "test",
        "n_examples": 5,
        "metrics": {
            "accuracy": {
                "mean": 0.9,
                "std": 0.1
            }
        },
        "latency_ms": {
            "mean": 1000.0
        },
        "tokens": {
            "avg_input_tokens": 100.0,
            "avg_output_tokens": 10.0
        }
    }
    
    row = summary_to_leaderboard_row(summary)
    
    # Should have default cost values
    assert row["total_cost_usd"] == 0.0
    assert row["avg_cost_usd"] == 0.0


def test_summary_to_leaderboard_row_scaledown_mode():
    """Test conversion identifies scaledown mode correctly."""
    summary = {
        "experiment_id": "hotpotqa_scaledown_openai_20251215_130312",
        "model": "openai:gpt-4o",
        "dataset": "hotpotqa",
        "n_examples": 10,
        "metrics": {
            "f1": {"mean": 0.75}
        },
        "latency_ms": {"mean": 2500.0},
        "tokens": {
            "avg_input_tokens": 200.0,  # Compressed
            "avg_output_tokens": 3.0
        },
        "cost": {
            "total_usd": 0.007,
            "mean_usd": 0.0007
        }
    }
    
    row = summary_to_leaderboard_row(summary)
    
    assert row["mode"] == "scaledown"
    assert row["compression_ratio"] is None  # Placeholder, needs baseline comparison
