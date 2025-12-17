"""Custom metrics collector for load testing.

This module provides thread-safe metrics tracking for Locust load testing,
capturing per-request and aggregated statistics for compression operations.
"""
import logging
import statistics
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Per-request metrics data."""
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    context_length: int
    success: bool
    latency_ms: float
    compression_rate: str
    compression_model: str


class MetricsCollector:
    """Thread-safe metrics collector for load testing.
    
    Collects per-request metrics and provides aggregated statistics.
    Uses threading.Lock() for thread-safe operations across concurrent
    Locust users.
    
    Tracks:
        - Per-request: tokens, compression ratio, context length, latency, success
        - Aggregated: mean/median/p95/p99, success rate, total requests
    
    Attributes:
        metrics: List of all recorded request metrics
        total_requests: Total number of requests recorded
        successful_requests: Number of successful requests
        failed_requests: Number of failed requests
    """
    
    def __init__(self):
        """Initialize metrics collector with thread-safe lock."""
        self.metrics: List[RequestMetrics] = []
        self._lock = threading.Lock()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
    
    def record(
        self,
        original_tokens: int,
        compressed_tokens: int,
        compression_ratio: float,
        context_length: int,
        success: bool,
        latency_ms: float,
        compression_rate: str = "auto",
        compression_model: str = "unknown",
    ) -> None:
        """Record metrics for a single request (thread-safe).
        
        Args:
            original_tokens: Original prompt token count
            compressed_tokens: Compressed prompt token count
            compression_ratio: Compression ratio (0.0-1.0)
            context_length: Context length in characters
            success: Whether request succeeded
            latency_ms: API latency in milliseconds
            compression_rate: Compression rate used ("auto" or float)
            compression_model: Model used for compression
        """
        metric = RequestMetrics(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            context_length=context_length,
            success=success,
            latency_ms=latency_ms,
            compression_rate=str(compression_rate),
            compression_model=compression_model,
        )
        
        with self._lock:
            self.metrics.append(metric)
            self.total_requests += 1
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
    
    def get_stats(self) -> Dict:
        """Get aggregated statistics (thread-safe).
        
        Returns:
            Dictionary with aggregated statistics including:
                - total_requests, successful_requests, failed_requests
                - success_rate (%)
                - For each numeric metric: mean, median, p95, p99, min, max
        """
        with self._lock:
            if not self.metrics:
                return {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "success_rate": 0.0,
                }
            
            # Filter successful requests for numeric stats
            successful = [m for m in self.metrics if m.success]
            
            if not successful:
                return {
                    "total_requests": self.total_requests,
                    "successful_requests": self.successful_requests,
                    "failed_requests": self.failed_requests,
                    "success_rate": 0.0,
                }
            
            # Calculate statistics for each numeric metric
            def calc_stats(values: List[float]) -> Dict:
                """Calculate statistics for a list of values."""
                if not values:
                    return {}
                
                return {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "p95": np.percentile(values, 95),
                    "p99": np.percentile(values, 99),
                    "min": min(values),
                    "max": max(values),
                }
            
            # Extract values
            latencies = [m.latency_ms for m in successful]
            compression_ratios = [m.compression_ratio for m in successful]
            original_tokens = [m.original_tokens for m in successful]
            compressed_tokens = [m.compressed_tokens for m in successful]
            context_lengths = [m.context_length for m in successful]
            
            # Calculate token savings
            token_savings = [
                orig - comp for orig, comp in zip(original_tokens, compressed_tokens)
            ]
            
            success_rate = (self.successful_requests / self.total_requests) * 100
            
            return {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": success_rate,
                "latency_ms": calc_stats(latencies),
                "compression_ratio": calc_stats(compression_ratios),
                "original_tokens": calc_stats(original_tokens),
                "compressed_tokens": calc_stats(compressed_tokens),
                "token_savings": calc_stats(token_savings),
                "context_length": calc_stats(context_lengths),
            }
    
    def export_to_json(self) -> Dict:
        """Export all metrics to JSON-serializable dictionary.
        
        Returns:
            Dictionary with aggregated stats (ready for JSON export)
        """
        return self.get_stats()
    
    def export_to_csv(self) -> str:
        """Export aggregated stats to CSV format.
        
        Returns:
            CSV string with aggregated statistics
        """
        stats = self.get_stats()
        
        # Build CSV header and rows
        lines = []
        
        # Summary row
        lines.append("metric,value")
        lines.append(f"total_requests,{stats['total_requests']}")
        lines.append(f"successful_requests,{stats['successful_requests']}")
        lines.append(f"failed_requests,{stats['failed_requests']}")
        lines.append(f"success_rate,{stats['success_rate']:.2f}")
        
        # Detailed stats for each metric
        for metric_name in ["latency_ms", "compression_ratio", "original_tokens", 
                            "compressed_tokens", "token_savings", "context_length"]:
            if metric_name in stats and isinstance(stats[metric_name], dict):
                metric_stats = stats[metric_name]
                for stat_name, value in metric_stats.items():
                    lines.append(f"{metric_name}_{stat_name},{value:.2f}")
        
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Reset all collected metrics (thread-safe).
        
        Useful for starting a new test run without recreating the collector.
        """
        with self._lock:
            self.metrics.clear()
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0

