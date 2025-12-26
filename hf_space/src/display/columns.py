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


# Column display name mappings for user-friendly headers
COLUMN_DISPLAY_NAMES = {
    # Fixed columns
    "model": "Model",
    "dataset": "Dataset",
    "mode": "Mode",
    "n_examples": "Examples",
    
    # Performance columns
    "avg_latency_ms": "Avg Latency (ms)",
    "latency_p50": "Latency P50 (ms)",
    "latency_p95": "Latency P95 (ms)",
    
    # Token columns
    "avg_input_tokens": "Avg Input Tokens",
    "avg_output_tokens": "Avg Output Tokens",
    "total_input_tokens": "Total Input Tokens",
    "total_output_tokens": "Total Output Tokens",
    
    # Cost columns
    "total_cost_usd": "Total Cost (USD)",
    "avg_cost_usd": "Avg Cost (USD)",
    "cost_p50_usd": "Cost P50 (USD)",
    "cost_p95_usd": "Cost P95 (USD)",
    
    # Compression columns
    "compression_ratio": "Compression Ratio",
    
    # Set metrics (ETHIC dataset)
    "set_exact_match": "Set EM",
    "set_precision": "Set Precision",
    "set_recall": "Set Recall",
    "set_f1": "Set F1",
    
    # Metadata columns
    "experiment_id": "Experiment ID",
    "timestamp": "Timestamp",
    "uploaded_at": "Uploaded At",
}


def get_display_name(column_name: str) -> str:
    """Get user-friendly display name for a column.
    
    Args:
        column_name: Internal column name.
        
    Returns:
        User-friendly display name.
    """
    # Check if we have an explicit mapping
    if column_name in COLUMN_DISPLAY_NAMES:
        return COLUMN_DISPLAY_NAMES[column_name]
    
    # Handle metric columns dynamically
    # exact_match_mean -> Exact Match (Mean)
    # f1_std -> F1 (Std)
    # bertscore_f1_mean -> BERTScore F1 (Mean)
    
    if "_mean" in column_name or "_std" in column_name:
        base_name = column_name.replace("_mean", "").replace("_std", "")
        suffix = "Mean" if "_mean" in column_name else "Std"
        
        # Format base name
        display = base_name.replace("_", " ").title()
        
        # Special case formatting
        if "f1" in display.lower() and "bertscore" not in display.lower():
            display = display.replace("F1", "F1")
        if "bertscore" in display.lower():
            display = display.replace("Bertscore", "BERTScore")
        
        return f"{display} ({suffix})"
    
    # Default: title case with spaces
    return column_name.replace("_", " ").title()


# Fixed columns that appear in all results
FIXED_COLUMNS = [
    ColumnConfig("model", get_display_name("model"), "str", displayed_by_default=True, never_hidden=True),
    ColumnConfig("dataset", get_display_name("dataset"), "str", displayed_by_default=True, never_hidden=True),
    ColumnConfig("mode", get_display_name("mode"), "str", displayed_by_default=True),
    ColumnConfig("n_examples", get_display_name("n_examples"), "number", displayed_by_default=True),
]

# Performance columns
PERFORMANCE_COLUMNS = [
    ColumnConfig("avg_latency_ms", get_display_name("avg_latency_ms"), "number", displayed_by_default=True),
    ColumnConfig("latency_p50", get_display_name("latency_p50"), "number", displayed_by_default=False),
    ColumnConfig("latency_p95", get_display_name("latency_p95"), "number", displayed_by_default=False),
]

# Token columns
TOKEN_COLUMNS = [
    ColumnConfig("avg_input_tokens", get_display_name("avg_input_tokens"), "number", displayed_by_default=True),
    ColumnConfig("avg_output_tokens", get_display_name("avg_output_tokens"), "number", displayed_by_default=False),
]

# Cost columns
COST_COLUMNS = [
    ColumnConfig("total_cost_usd", get_display_name("total_cost_usd"), "number", displayed_by_default=False),
    ColumnConfig("avg_cost_usd", get_display_name("avg_cost_usd"), "number", displayed_by_default=True),
]

# Metadata columns
METADATA_COLUMNS = [
    ColumnConfig("experiment_id", get_display_name("experiment_id"), "str", displayed_by_default=False, hidden=True),
    ColumnConfig("timestamp", get_display_name("timestamp"), "str", displayed_by_default=False),
    ColumnConfig("uploaded_at", get_display_name("uploaded_at"), "str", displayed_by_default=False, hidden=True),
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
    display_name = get_display_name(metric_name)
    
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
