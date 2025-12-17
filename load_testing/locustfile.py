"""Main Locust entrypoint for ScaleDown API load testing.

This module defines the Locust User class and tasks for load testing
the ScaleDown compression API. It uses shared singleton instances of
DataSampler and MetricsCollector for efficiency.

Usage:
    # Web UI mode (default)
    locust -f load_testing/locustfile.py

    # Headless mode with specific parameters
    locust -f load_testing/locustfile.py --headless --users=50 --spawn-rate=5 --run-time=5m
"""
import logging
import random
from pathlib import Path

from locust import User, task, events, between
from locust.runners import MasterRunner

from load_testing.config import load_config
from load_testing.data_sampler import DataSampler
from load_testing.compression_client import compress_context_api
from load_testing.metrics_collector import MetricsCollector


logger = logging.getLogger(__name__)


# Shared singleton instances (class-level attributes)
# These will be initialized once and shared across all user instances
_config = None
_data_sampler = None
_metrics_collector = None


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize shared resources when Locust starts.
    
    This is called once per Locust process before any users are spawned.
    We initialize the config, data sampler, and metrics collector here
    so they're shared across all user instances.
    """
    global _config, _data_sampler, _metrics_collector
    
    # Only initialize on worker or standalone nodes (not on master)
    if isinstance(environment.runner, MasterRunner):
        logger.info("Running as master node - skipping data initialization")
        return
    
    logger.info("Initializing load test environment...")
    
    # Load configuration
    config_path = Path("load_testing/config.yaml")
    _config = load_config(str(config_path))
    logger.info(f"Loaded config from {config_path}")
    
    # Initialize data sampler
    _data_sampler = DataSampler(
        dataset_name=_config.data.dataset,
        split=_config.data.split,
        num_samples=_config.data.num_samples,
        seed=_config.data.seed,
    )
    logger.info(f"Initialized DataSampler with {_data_sampler.num_loaded} examples")
    
    # Initialize metrics collector
    _metrics_collector = MetricsCollector()
    logger.info("Initialized MetricsCollector")
    
    logger.info("Load test environment ready!")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Export custom metrics when test stops.
    
    This is called when the load test ends. We export our custom metrics
    to JSON and CSV files in the results directory.
    """
    global _metrics_collector, _config
    
    if _metrics_collector is None:
        logger.warning("MetricsCollector not initialized - skipping export")
        return
    
    logger.info("Exporting custom metrics...")
    
    # Export to JSON
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(_config.output.results_dir) / "summaries"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Export JSON summary
    json_path = results_dir / f"load_test_metrics_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(_metrics_collector.export_to_json(), f, indent=2)
    logger.info(f"Exported metrics to {json_path}")
    
    # Export CSV summary
    csv_path = results_dir / f"load_test_metrics_{timestamp}.csv"
    with open(csv_path, "w") as f:
        f.write(_metrics_collector.export_to_csv())
    logger.info(f"Exported metrics to {csv_path}")
    
    # Print summary to console
    stats = _metrics_collector.get_stats()
    logger.info(f"Load Test Summary:")
    logger.info(f"  Total requests: {stats['total_requests']}")
    logger.info(f"  Successful: {stats['successful_requests']}")
    logger.info(f"  Failed: {stats['failed_requests']}")
    logger.info(f"  Success rate: {stats['success_rate']:.2f}%")
    
    if stats['successful_requests'] > 0:
        logger.info(f"  Avg latency: {stats['latency_ms']['mean']:.2f}ms")
        logger.info(f"  P95 latency: {stats['latency_ms']['p95']:.2f}ms")
        logger.info(f"  Avg compression ratio: {stats['compression_ratio']['mean']:.2%}")
        logger.info(f"  Avg token savings: {stats['token_savings']['mean']:.0f} tokens")


class ScaleDownUser(User):
    """Locust User for testing ScaleDown compression API.
    
    This User class simulates real users making compression requests to
    the ScaleDown API. Each user:
    - Gets a random example from the dataset
    - Calls the ScaleDown compression API
    - Records custom metrics
    
    The wait_time is set to a small value (0.1-1s) to generate steady load.
    Adjust users and spawn rate via CLI to control overall load.
    """
    
    # Wait time between tasks (in seconds)
    # Using 'between' for random wait time to simulate realistic usage
    wait_time = between(0.1, 1.0)
    
    @task
    def compress_context_task(self):
        """Main task: compress a random context using ScaleDown API.
        
        This task:
        1. Gets a random example from the dataset
        2. Selects random compression rate and model (if multiple configured)
        3. Calls ScaleDown compression API
        4. Records metrics (including Locust's built-in stats via custom request)
        """
        # Get random example from dataset
        example = _data_sampler.get_random_example()
        context = example["context"]
        prompt = example["question"]
        
        # Randomly select compression rate and model if multiple configured
        compression_rate = random.choice(_config.compression.rates)
        compression_model = random.choice(_config.compression.models)
        
        # Create a descriptive name for Locust stats grouping
        # This groups requests by rate/model in Locust's statistics
        request_name = f"/compress (rate={compression_rate}, model={compression_model})"
        
        # Call compression API using our standalone function
        # We don't use self.client.post() because we need custom retry logic
        # Instead, we manually fire the request event for Locust stats
        result = compress_context_api(
            context=context,
            prompt=prompt,
            compression_rate=compression_rate,
            compression_model=compression_model,
            api_key=_config.scaledown.api_key,
            api_url=_config.scaledown.api_url,
            timeout=_config.scaledown.timeout,
            max_retries=_config.scaledown.max_retries,
            retry_delay=_config.scaledown.retry_delay,
        )
        
        # Fire Locust's request event for built-in statistics
        # This makes the request appear in Locust's web UI and CSV exports
        if result["success"]:
            self.environment.events.request.fire(
                request_type="POST",
                name=request_name,
                response_time=result["latency_ms"],
                response_length=result["metadata"]["compressed_prompt_tokens"],
                exception=None,
                context={},
            )
        else:
            # Fire failure event
            self.environment.events.request.fire(
                request_type="POST",
                name=request_name,
                response_time=result["latency_ms"],
                response_length=0,
                exception=Exception(result["error"]),
                context={},
            )
        
        # Record custom metrics in our MetricsCollector
        if result["success"]:
            _metrics_collector.record(
                original_tokens=result["metadata"]["original_prompt_tokens"],
                compressed_tokens=result["metadata"]["compressed_prompt_tokens"],
                compression_ratio=result["metadata"]["compression_ratio"],
                context_length=len(context),
                success=True,
                latency_ms=result["latency_ms"],
                compression_rate=str(compression_rate),
                compression_model=compression_model,
            )
        else:
            # Record failure (with zero tokens)
            _metrics_collector.record(
                original_tokens=0,
                compressed_tokens=0,
                compression_ratio=0.0,
                context_length=len(context),
                success=False,
                latency_ms=result["latency_ms"],
                compression_rate=str(compression_rate),
                compression_model=compression_model,
            )

