"""MLflow utilities: setup, configuration, and enable/disable control.

This module provides centralized MLflow setup with on/off control via config or CLI.
Key features:
- Setup MLflow tracking URI and experiment
- Enable auto-tracing for OpenAI and Gemini
- Global enable/disable flag to gate all MLflow operations
- Context manager and decorator helpers for tracing
- Robust retry configuration for trace logging
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Global flag to track if MLflow is enabled for this session
_mlflow_enabled = False


def is_mlflow_enabled() -> bool:
    """Check if MLflow is enabled for the current session.
    
    Returns:
        True if MLflow was successfully initialized and enabled.
    """
    return _mlflow_enabled


def setup_mlflow(
    enabled: bool = True,
    tracking_uri: str = "http://localhost:5000",
    experiment_name: str = "scaledown-benchmarking",
) -> bool:
    """Initialize MLflow if enabled.
    
    Sets up MLflow tracking URI, experiment, and enables auto-tracing
    for supported providers (OpenAI, Gemini).
    
    Args:
        enabled: Whether to enable MLflow. If False, no-op.
        tracking_uri: MLflow tracking server URI.
        experiment_name: Name of the MLflow experiment.
        
    Returns:
        True if MLflow was successfully initialized, False otherwise.
    """
    global _mlflow_enabled
    
    if not enabled:
        _mlflow_enabled = False
        logger.info("MLflow disabled by configuration")
        return False
    
    try:
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "5")  
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "120")  
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "2") 
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_JITTER", "1.0")  
        os.environ.setdefault("MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT", "300")  
        os.environ.setdefault("MLFLOW_ASYNC_TRACE_LOGGING_MAX_WORKERS", "3")  
        
        import mlflow
        
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")
        
        # Set experiment (will create if doesn't exist)
        try:
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment set to: {experiment_name}")
        except Exception as e:
            logger.error(f"Failed to set MLflow experiment '{experiment_name}': {e}")
            logger.info("Attempting to create experiment explicitly...")
            try:
                experiment_id = mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
                logger.info(f"Created MLflow experiment '{experiment_name}' with ID: {experiment_id}")
            except Exception as create_error:
                logger.error(f"Failed to create experiment: {create_error}")
                raise
        
        # For each supported provider, enable auto-tracing. Add this for new providers as needed. 

        # Enable auto-tracing for OpenAI
        try:
            mlflow.openai.autolog()
            logger.info("MLflow OpenAI auto-tracing enabled")
        except Exception as e:
            logger.warning(f"Failed to enable OpenAI auto-tracing: {e}")
        
        # Enable auto-tracing for Gemini
        try:
            mlflow.gemini.autolog()
            logger.info("MLflow Gemini auto-tracing enabled")
        except Exception as e:
            logger.warning(f"Failed to enable Gemini auto-tracing: {e}")
        
        _mlflow_enabled = True
        logger.info(
            f"MLflow initialized successfully: tracking_uri={tracking_uri}, "
            f"experiment={experiment_name}"
        )
        return True
        
    except ImportError:
        logger.warning("MLflow not installed. Run: pip install mlflow>=3.0")
        _mlflow_enabled = False
        return False
    except Exception as e:
        logger.error(f"Failed to initialize MLflow: {e}")
        _mlflow_enabled = False
        return False


def disable_mlflow() -> None:
    """Disable MLflow for the current session.
    
    Disables auto-tracing and sets the global flag to False.
    """
    global _mlflow_enabled
    
    if not _mlflow_enabled:
        return
    
    try:
        import mlflow
        
        # Disable auto-tracing
        try:
            mlflow.openai.autolog(disable=True)
        except Exception:
            pass
        
        try:
            mlflow.gemini.autolog(disable=True)
        except Exception:
            pass
        
        logger.info("MLflow disabled")
        
    except ImportError:
        pass
    
    _mlflow_enabled = False


def start_run(
    run_name: str,
    tags: Optional[dict] = None,
    description: Optional[str] = None,
):
    """Start an MLflow run if MLflow is enabled.
    
    Returns a context manager that can be used with `with` statement.
    If MLflow is disabled, returns a no-op context manager.
    
    Args:
        run_name: Name for the MLflow run.
        tags: Optional dict of tags to set on the run.
        description: Optional description for the run.
        
    Returns:
        MLflow run context manager or no-op context manager.
    """
    if not _mlflow_enabled:
        return _NoOpContextManager()
    
    try:
        import mlflow
        
        return mlflow.start_run(
            run_name=run_name,
            tags=tags,
            description=description,
        )
    except Exception as e:
        logger.warning(f"Failed to start MLflow run: {e}")
        return _NoOpContextManager()


def log_params(params: dict) -> None:
    """Log parameters to MLflow if enabled.
    
    Args:
        params: Dictionary of parameters to log.
    """
    if not _mlflow_enabled:
        return
    
    try:
        import mlflow
        mlflow.log_params(params)
    except Exception as e:
        logger.warning(f"Failed to log MLflow params: {e}")


def log_metrics(metrics: dict, step: Optional[int] = None) -> None:
    """Log metrics to MLflow if enabled.
    
    Args:
        metrics: Dictionary of metrics to log.
        step: Optional step number for the metrics.
    """
    if not _mlflow_enabled:
        return
    
    try:
        import mlflow
        mlflow.log_metrics(metrics, step=step)
    except Exception as e:
        logger.warning(f"Failed to log MLflow metrics: {e}")


def log_artifact(local_path: str, artifact_path: Optional[str] = None) -> None:
    """Log an artifact to MLflow if enabled.
    
    Args:
        local_path: Path to the local file to log.
        artifact_path: Optional path within the artifact store.
    """
    if not _mlflow_enabled:
        return
    
    try:
        import mlflow
        mlflow.log_artifact(local_path, artifact_path=artifact_path)
    except Exception as e:
        logger.warning(f"Failed to log MLflow artifact: {e}")


def set_tags(tags: dict) -> None:
    """Set tags on the current MLflow run if enabled.
    
    Args:
        tags: Dictionary of tags to set.
    """
    if not _mlflow_enabled:
        return
    
    try:
        import mlflow
        mlflow.set_tags(tags)
    except Exception as e:
        logger.warning(f"Failed to set MLflow tags: {e}")


def start_span(name: str, span_type: Optional[str] = None):
    """Start an MLflow span if enabled.
    
    Returns a context manager for manual span tracing.
    If MLflow is disabled, returns a no-op context manager.
    
    Args:
        name: Name of the span.
        span_type: Optional span type (e.g., "CHAIN", "LLM", "TOOL").
        
    Returns:
        MLflow span context manager or no-op context manager.
    """
    if not _mlflow_enabled:
        return _NoOpSpan()
    
    try:
        import mlflow
        
        kwargs = {"name": name}
        if span_type:
            kwargs["span_type"] = span_type
        
        return mlflow.start_span(**kwargs)
    except Exception as e:
        logger.warning(f"Failed to start MLflow span: {e}")
        return _NoOpSpan()


def get_current_active_span():
    """Get the current active MLflow span if one exists.
    
    Returns:
        The current active span, or None if no span is active or MLflow is disabled.
    """
    if not _mlflow_enabled:
        return None
    
    try:
        import mlflow
        return mlflow.get_current_active_span()
    except Exception as e:
        logger.warning(f"Failed to get current MLflow span: {e}")
        return None


def set_span_attribute(key: str, value) -> None:
    """Set an attribute on the current active span if one exists.
    
    This is useful for adding metadata like gold/reference answers to auto-traced spans.
    
    Args:
        key: The attribute key.
        value: The attribute value.
    """
    if not _mlflow_enabled:
        return
    
    try:
        span = get_current_active_span()
        if span:
            span.set_attribute(key, value)
    except Exception as e:
        logger.warning(f"Failed to set span attribute: {e}")


class _NoOpContextManager:
    """No-op context manager for when MLflow is disabled."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class _NoOpSpan:
    """No-op span for when MLflow is disabled."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
    
    def set_inputs(self, inputs):
        pass
    
    def set_outputs(self, outputs):
        pass
    
    def set_attribute(self, key, value):
        pass
