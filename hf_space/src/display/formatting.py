"""Formatting utilities for leaderboard display."""
from typing import Any
import pandas as pd


def format_number(value: Any, decimals: int = 4) -> str:
    """Format numeric value for display."""
    if pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def format_cost(value: Any) -> str:
    """Format cost value with dollar sign."""
    if pd.isna(value):
        return "-"
    if isinstance(value, (int, float)):
        return f"${value:.4f}"
    return str(value)


def format_latency(value: Any) -> str:
    """Format latency in milliseconds."""
    if pd.isna(value):
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.1f}ms"
    return str(value)


def format_percentage(value: Any) -> str:
    """Format percentage value."""
    if pd.isna(value):
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.1f}%"
    return str(value)


def format_delta(value: Any, higher_is_better: bool = True) -> str:
    """Format delta value with +/- prefix.
    
    Args:
        value: Delta value.
        higher_is_better: If True, positive deltas are good (green).
    """
    if pd.isna(value):
        return "-"
    if isinstance(value, (int, float)):
        prefix = "+" if value > 0 else ""
        return f"{prefix}{value:.4f}"
    return str(value)


def styled_error(error: str) -> str:
    """Style error message in red."""
    return f"<p style='color: red; font-size: 18px; text-align: center;'>{error}</p>"


def styled_message(message: str) -> str:
    """Style success message in green."""
    return f"<p style='color: green; font-size: 18px; text-align: center;'>{message}</p>"


def has_no_nan_values(df: pd.DataFrame, columns: list) -> pd.Series:
    """Check which rows have no NaN values in specified columns."""
    existing_cols = [c for c in columns if c in df.columns]
    if not existing_cols:
        return pd.Series([True] * len(df))
    return df[existing_cols].notna().all(axis=1)
