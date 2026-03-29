"""LangSmith utilities: setup, configuration, and enable/disable control.

This module provides centralized LangSmith setup with on/off control via
config or CLI. Mirrors the structure of mlflow_utils.py so both tracing
backends are toggled and used identically.

Key features:
- Setup LangSmith project and client
- Global enable/disable flag to gate all operations
- Context managers for experiment runs and per-example child runs
- Manual LLM call tracing (for non-LangChain clients)
- Feedback logging for per-example metric scores
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Global state
_langsmith_enabled: bool = False
_langsmith_client = None  # langsmith.Client instance
_langsmith_project: str = "scalebench"


def is_langsmith_enabled() -> bool:
    """Check if LangSmith is enabled for the current session."""
    return _langsmith_enabled


def setup_langsmith(
    enabled: bool = True,
    project_name: str = "ScaleDown",
    api_key: Optional[str] = None,
) -> bool:
    """Initialize LangSmith if enabled.

    Args:
        enabled: Whether to enable LangSmith. If False, no-op.
        project_name: LangSmith project name.
        api_key: API key. Falls back to LANGSMITH_API_KEY / LANGCHAIN_API_KEY
                 env vars if None (handled by the langsmith SDK automatically).

    Returns:
        True if successfully initialized, False otherwise.
    """
    global _langsmith_enabled, _langsmith_client, _langsmith_project

    if not enabled:
        _langsmith_enabled = False
        logger.info("LangSmith disabled by configuration")
        return False

    try:
        import langsmith

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key

        _langsmith_client = langsmith.Client(**kwargs)
        _langsmith_project = project_name

        # Light connectivity check — warn only, don't fail init
        try:
            list(_langsmith_client.list_projects(limit=1))
        except Exception as e:
            logger.warning(
                f"LangSmith connectivity check failed: {e}. "
                "Tracing may not work correctly."
            )

        _langsmith_enabled = True
        logger.info(
            f"LangSmith initialized: project={project_name}"
        )
        return True

    except ImportError:
        logger.warning(
            "langsmith not installed. Run: pip install langsmith>=0.1.147"
        )
        _langsmith_enabled = False
        return False
    except Exception as e:
        logger.error(f"Failed to initialize LangSmith: {e}")
        _langsmith_enabled = False
        return False


def disable_langsmith() -> None:
    """Disable LangSmith for the current session."""
    global _langsmith_enabled, _langsmith_client
    _langsmith_enabled = False
    _langsmith_client = None
    logger.info("LangSmith disabled")


def start_run(run_name: str, tags: Optional[Dict[str, str]] = None):
    """Start a LangSmith experiment-level run.

    Mirrors mlflow_utils.start_run(). Returns a context manager.
    Run name is suffixed with a timestamp for uniqueness.

    Args:
        run_name: Base name for the run.
        tags: Optional dict of tags (values converted to strings).

    Returns:
        _LangSmithRun context manager, or _NoOpRun if disabled.
    """
    if not _langsmith_enabled:
        return _NoOpRun()

    try:
        from langsmith import RunTree

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_name = f"{run_name}_{timestamp}"

        # RunTree.tags is List[str]; encode dict tags as "key:value" strings
        tag_list = (
            [f"{k}:{v}" for k, v in tags.items()] if tags else []
        )

        run_tree = RunTree(
            name=unique_name,
            run_type="chain",
            project_name=_langsmith_project,
            tags=tag_list,
            start_time=datetime.now(timezone.utc),
        )
        run_tree.post()
        return _LangSmithRun(run_tree)

    except Exception as e:
        logger.warning(f"Failed to start LangSmith run: {e}")
        return _NoOpRun()


def start_child_run(parent_run_tree, name: str):
    """Start a per-example child run under a parent RunTree.

    Mirrors mlflow_utils.start_span(). Returns a context manager.

    Args:
        parent_run_tree: The parent langsmith.RunTree
            (from _LangSmithRun.run_tree). Pass None when disabled — no-op.
        name: Name for the child run.

    Returns:
        _LangSmithChildRun context manager, or _NoOpChildRun if disabled.
    """
    if not _langsmith_enabled or parent_run_tree is None:
        return _NoOpChildRun()

    try:
        now = datetime.now(timezone.utc)
        child = parent_run_tree.create_child(
            name=name,
            run_type="chain",
        )
        child.start_time = now
        child.post()
        return _LangSmithChildRun(child)

    except Exception as e:
        logger.warning(f"Failed to start LangSmith child run: {e}")
        return _NoOpChildRun()


def log_llm_call(
    parent_child_run,
    generation_input,
    output,
    model_name: str,
) -> None:
    """Log a single LLM call as a nested 'llm' run under the example child run.

    Captures the full prompt (system + user messages) and the raw model
    response along with token counts, latency, and cost.

    Args:
        parent_child_run: The langsmith.RunTree from
            _LangSmithChildRun.child_run. Safe to call with None — no-op.
        generation_input: GenerationInput instance with prompt fields.
        output: ModelOutput instance with text/token/cost/latency fields.
        model_name: Human-readable model identifier (e.g. "openai:gpt-4o").
    """
    if not _langsmith_enabled or parent_child_run is None:
        return

    try:
        messages = []
        if generation_input.system_prompt:
            messages.append(
                {"role": "system", "content": generation_input.system_prompt}
            )
        user_content = generation_input.user_prompt or ""
        if generation_input.context:
            user_content = f"{generation_input.context}\n\n{user_content}"
        messages.append({"role": "user", "content": user_content})

        end_time = datetime.now(timezone.utc)
        latency_s = (output.latency_ms or 0) / 1000.0
        start_time = end_time - timedelta(seconds=latency_s)

        llm_run = parent_child_run.create_child(
            name=f"llm_{model_name}",
            run_type="llm",
            inputs={"messages": messages},
        )
        llm_run.start_time = start_time
        llm_run.post()
        llm_run.end(
            outputs={"generations": [{"text": output.text or ""}]},
            metadata={
                "model": model_name,
                "input_tokens": output.input_tokens,
                "output_tokens": output.output_tokens,
                "latency_ms": output.latency_ms,
                "cost_usd": output.cost_usd,
            },
            end_time=end_time,
        )
        llm_run.patch()

    except Exception as e:
        logger.warning(f"Failed to log LangSmith LLM call: {e}")


def log_feedback(run_id: Any, key: str, score: float) -> None:
    """Log a metric score as LangSmith feedback on a run.

    Equivalent to mlflow.log_metrics() but at the per-run level.

    Args:
        run_id: The run UUID (from child_run.id).
        key: Metric name (e.g. "rouge1", "f1", "exact_match").
        score: Numeric score in [0, 1].
    """
    if not _langsmith_enabled or _langsmith_client is None:
        return

    try:
        _langsmith_client.create_feedback(
            run_id=run_id,
            key=key,
            score=score,
        )
    except Exception as e:
        logger.warning(f"Failed to log LangSmith feedback '{key}': {e}")


def flush_traces() -> None:
    """No-op for API parity with mlflow_utils.

    LangSmith's RunTree.patch() is synchronous, so all data is already
    persisted by the time context managers exit.
    """
    logger.debug("LangSmith flush_traces called (no-op: already synchronous)")


# ---------------------------------------------------------------------------
# Internal context manager wrappers
# ---------------------------------------------------------------------------


class _LangSmithRun:
    """Context manager wrapping a top-level experiment RunTree.

    Metadata and outputs are accumulated and flushed via end() on context exit.
    """

    def __init__(self, run_tree):
        self.run_tree = run_tree
        self._metadata: dict = {}
        self._outputs: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.run_tree.end(
                outputs=self._outputs if self._outputs else None,
                metadata=self._metadata if self._metadata else None,
                end_time=datetime.now(timezone.utc),
            )
            self.run_tree.patch()
        except Exception as e:
            logger.warning(f"Failed to end LangSmith run: {e}")
        return False

    def log_params(self, params: dict) -> None:
        """Accumulate experiment-level parameters — flushed on context exit."""
        self._metadata.update(params)

    def set_outputs(self, outputs: dict) -> None:
        """Accumulate experiment-level outputs — flushed on context exit."""
        self._outputs.update(outputs)


class _LangSmithChildRun:
    """Context manager wrapping a per-example child RunTree.

    Outputs and metadata are accumulated and flushed together in __exit__
    via end(), which is the correct way to finalize a RunTree without
    overwriting data from earlier patch() calls.
    """

    def __init__(self, child_run):
        self.child_run = child_run
        self._inputs: dict = {}
        self._outputs: dict = {}
        self._metadata: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._inputs:
                self.child_run.inputs = self._inputs
            self.child_run.end(
                outputs=self._outputs if self._outputs else None,
                metadata=self._metadata if self._metadata else None,
                end_time=datetime.now(timezone.utc),
            )
            self.child_run.patch()
        except Exception as e:
            logger.warning(f"Failed to end LangSmith child run: {e}")
        return False

    def set_inputs(self, inputs: dict) -> None:
        """Accumulate inputs — flushed to LangSmith on context exit."""
        self._inputs.update(inputs)

    def set_outputs(self, outputs: dict) -> None:
        """Accumulate outputs — flushed to LangSmith on context exit."""
        self._outputs.update(outputs)

    def set_metadata(self, metadata: dict) -> None:
        """Accumulate metadata — flushed to LangSmith on context exit."""
        self._metadata.update(metadata)


class _NoOpRun:
    """No-op context manager for when LangSmith is disabled."""

    run_tree = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def log_params(self, params: dict) -> None:
        pass

    def set_outputs(self, outputs: dict) -> None:
        pass


class _NoOpChildRun:
    """No-op child run for when LangSmith is disabled."""

    child_run = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def set_inputs(self, inputs: dict) -> None:
        pass

    def set_outputs(self, outputs: dict) -> None:
        pass

    def set_metadata(self, metadata: dict) -> None:
        pass
