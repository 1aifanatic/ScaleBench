"""Column definitions for leaderboard table.

Uses dynamic column discovery to handle variable metrics across tasks.
"""
from dataclasses import dataclass
from typing import List, Set


@dataclass
class ColumnConfig:
    """Configuration for a leaderboard column."""
    name: str
    display_name: str
    col_type: str  # 'str', 'number', 'markdown'
    displayed_by_default: bool = True
    hidden: bool = False
    never_hidden: bool = False


# Fixed columns that appear in all results
FIXED_COLUMNS = [
    ColumnConfig("model", "Model", "str", displayed_by_default=True, never_hidden=True),
    ColumnConfig("dataset", "Dataset", "str", displayed_by_default=True, never_hidden=True),
    ColumnConfig("task", "Task", "str", displayed_by_default=True),
    ColumnConfig("mode", "Mode", "str", displayed_by_default=True),
    ColumnConfig("n_examples", "N", "number", displayed_by_default=True),
]

# Performance columns
PERFORMANCE_COLUMNS = [
    ColumnConfig("avg_latency_ms", "Latency (ms)", "number", displayed_by_default=True),
    ColumnConfig("latency_p50", "Latency P50", "number", displayed_by_default=False),
    ColumnConfig("latency_p95", "Latency P95", "number", displayed_by_default=False),
]

# Token columns
TOKEN_COLUMNS = [
    ColumnConfig("avg_input_tokens", "Avg Input Tokens", "number", displayed_by_default=True),
    ColumnConfig("avg_output_tokens", "Avg Output Tokens", "number", displayed_by_default=False),
]

# Cost columns
COST_COLUMNS = [
    ColumnConfig("total_cost_usd", "Total Cost ($)", "number", displayed_by_default=False),
    ColumnConfig("avg_cost_usd", "Avg Cost ($)", "number", displayed_by_default=True),
]

# Metadata columns
METADATA_COLUMNS = [
    ColumnConfig("experiment_id", "Experiment ID", "str", displayed_by_default=False, hidden=True),
    ColumnConfig("timestamp", "Timestamp", "str", displayed_by_default=False),
    ColumnConfig("uploaded_at", "Uploaded", "str", displayed_by_default=False, hidden=True),
]


def get_all_fixed_columns() -> List[ColumnConfig]:
    """Get all fixed (non-metric) columns."""
    return FIXED_COLUMNS + PERFORMANCE_COLUMNS + TOKEN_COLUMNS + COST_COLUMNS + METADATA_COLUMNS


def create_metric_column(metric_name: str) -> ColumnConfig:
    """Create column config for a dynamic metric.
    
    Args:
        metric_name: Full metric column name (e.g., 'exact_match_mean').
        
    Returns:
        ColumnConfig for the metric.
    """
    # Convert metric_name to display name
    # exact_match_mean -> Exact Match
    # f1_mean -> F1
    # bertscore_f1_mean -> BERTScore F1
    base_name = metric_name.replace("_mean", "").replace("_std", " (std)")
    display_name = base_name.replace("_", " ").title()
    
    # Special case formatting
    if "f1" in display_name.lower():
        display_name = display_name.replace("F1", "F1")
    if "bertscore" in display_name.lower():
        display_name = display_name.replace("Bertscore", "BERTScore")
    
    # Mean columns shown by default, std columns hidden
    displayed = "_mean" in metric_name
    
    return ColumnConfig(
        name=metric_name,
        display_name=display_name,
        col_type="number",
        displayed_by_default=displayed,
        hidden=not displayed
    )


def build_column_config(metric_cols: Set[str]) -> List[ColumnConfig]:
    """Build complete column configuration including dynamic metrics.
    
    Args:
        metric_cols: Set of metric column names discovered from data.
        
    Returns:
        List of all column configs in display order.
    """
    columns = []
    
    # Add fixed columns first
    columns.extend(FIXED_COLUMNS)
    
    # Add dynamic metric columns (sorted for consistency)
    for metric_name in sorted(metric_cols):
        columns.append(create_metric_column(metric_name))
    
    # Add performance, token, cost columns
    columns.extend(PERFORMANCE_COLUMNS)
    columns.extend(TOKEN_COLUMNS)
    columns.extend(COST_COLUMNS)
    
    # Add metadata columns last
    columns.extend(METADATA_COLUMNS)
    
    return columns


def get_column_names(columns: List[ColumnConfig], include_hidden: bool = False) -> List[str]:
    """Get column names for DataFrame selection."""
    if include_hidden:
        return [c.name for c in columns]
    return [c.name for c in columns if not c.hidden]


def get_default_columns(columns: List[ColumnConfig]) -> List[str]:
    """Get columns to display by default."""
    return [c.name for c in columns if c.displayed_by_default]


def get_never_hidden_columns(columns: List[ColumnConfig]) -> List[str]:
    """Get columns that can never be hidden."""
    return [c.name for c in columns if c.never_hidden]


def get_column_types(columns: List[ColumnConfig]) -> List[str]:
    """Get column types for Gradio datatype parameter."""
    return [c.col_type for c in columns if not c.hidden]
