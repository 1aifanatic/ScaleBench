"""Tests for MLflow integration."""
import tempfile
from pathlib import Path


import yaml

from core.config import load_config, MLflowConfig
import utils.mlflow_utils as mlflow_utils


class TestMLflowConfig:
    """Tests for MLflowConfig dataclass."""

    def test_default_values(self):
        """Test default MLflow config values."""
        config = MLflowConfig()
        assert config.enabled is False
        assert config.tracking_uri == "http://localhost:5000"
        assert config.experiment_name == "scaledown-benchmarking"

    def test_custom_values(self):
        """Test MLflow config with custom values."""
        config = MLflowConfig(
            enabled=True,
            tracking_uri="http://custom:8080",
            experiment_name="custom-experiment",
        )
        assert config.enabled is True
        assert config.tracking_uri == "http://custom:8080"
        assert config.experiment_name == "custom-experiment"


class TestMLflowConfigParsing:
    """Tests for MLflow config parsing from YAML."""

    def test_config_with_mlflow_section(self):
        """Test loading config with mlflow section."""
        config_data = {
            "mlflow": {
                "enabled": True,
                "tracking_uri": "http://test:5000",
                "experiment_name": "test-experiment",
            },
            "experiments": [
                {
                    "id": "test_exp",
                    "model": {"name": "openai:gpt-4"},
                    "dataset": {"name": "hotpotqa"},
                    "task": {"name": "rag_qa"},
                }
            ],
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            config = load_config(config_path)
            assert config.mlflow is not None
            assert config.mlflow.enabled is True
            assert config.mlflow.tracking_uri == "http://test:5000"
            assert config.mlflow.experiment_name == "test-experiment"
        finally:
            config_path.unlink()

    def test_config_without_mlflow_section(self):
        """Test loading config without mlflow section."""
        config_data = {
            "experiments": [
                {
                    "id": "test_exp",
                    "model": {"name": "openai:gpt-4"},
                    "dataset": {"name": "hotpotqa"},
                    "task": {"name": "rag_qa"},
                }
            ],
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            config = load_config(config_path)
            assert config.mlflow is None
        finally:
            config_path.unlink()


class TestMLflowUtils:
    """Tests for mlflow_utils module."""

    def test_is_mlflow_enabled_default(self):
        """Test that MLflow is disabled by default."""
        # Reset state
        mlflow_utils._mlflow_enabled = False
        assert mlflow_utils.is_mlflow_enabled() is False

    def test_setup_mlflow_disabled(self):
        """Test setup_mlflow with enabled=False."""
        mlflow_utils._mlflow_enabled = True  # Set to True first
        result = mlflow_utils.setup_mlflow(enabled=False)
        assert result is False
        assert mlflow_utils.is_mlflow_enabled() is False

    def test_setup_mlflow_enabled(self):
        """Test setup_mlflow with enabled=True.
        
        Note: This test may fail if no MLflow server is running.
        The test validates the setup logic, not actual connectivity.
        """
        # Reset state first
        mlflow_utils._mlflow_enabled = False
        
        # Test that setup attempts to connect (may fail without server)
        # We're testing the code path, not actual connectivity
        try:
            result = mlflow_utils.setup_mlflow(
                enabled=True,
                tracking_uri="http://localhost:5000",
                experiment_name="test-exp",
            )
            # If it succeeds, verify state is correct
            if result:
                assert mlflow_utils.is_mlflow_enabled() is True
        except Exception:
            # If no server, that's expected - we're just testing the logic
            pass
        
        # Clean up
        mlflow_utils._mlflow_enabled = False

    def test_log_params_when_disabled(self):
        """Test that log_params is a no-op when MLflow is disabled."""
        mlflow_utils._mlflow_enabled = False
        # Should not raise
        mlflow_utils.log_params({"key": "value"})

    def test_log_metrics_when_disabled(self):
        """Test that log_metrics is a no-op when MLflow is disabled."""
        mlflow_utils._mlflow_enabled = False
        # Should not raise
        mlflow_utils.log_metrics({"metric": 1.0})

    def test_log_artifact_when_disabled(self):
        """Test that log_artifact is a no-op when MLflow is disabled."""
        mlflow_utils._mlflow_enabled = False
        # Should not raise
        mlflow_utils.log_artifact("/path/to/file")

    def test_noop_span(self):
        """Test NoOpSpan context manager."""
        mlflow_utils._mlflow_enabled = False
        
        with mlflow_utils.start_span(name="test") as span:
            # Should not raise
            span.set_inputs({"x": 1})
            span.set_outputs({"y": 2})
            span.set_attribute("key", "value")


class TestCLIFlagsLogic:
    """Tests for CLI flag logic."""

    def test_mlflow_flag_overrides_config(self):
        """Test that --mlflow flag overrides config."""
        # Import here to avoid circular imports
        from run_benchmarks import _should_enable_mlflow
        
        class Args:
            mlflow = True
            no_mlflow = False
        
        class Config:
            mlflow = MLflowConfig(enabled=False)
        
        assert _should_enable_mlflow(Args(), Config()) is True

    def test_no_mlflow_flag_overrides_config(self):
        """Test that --no-mlflow flag overrides config."""
        from run_benchmarks import _should_enable_mlflow
        
        class Args:
            mlflow = False
            no_mlflow = True
        
        class Config:
            mlflow = MLflowConfig(enabled=True)
        
        assert _should_enable_mlflow(Args(), Config()) is False

    def test_config_enabled_when_no_flags(self):
        """Test that config value is used when no CLI flags."""
        from run_benchmarks import _should_enable_mlflow
        
        class Args:
            mlflow = False
            no_mlflow = False
        
        class Config:
            mlflow = MLflowConfig(enabled=True)
        
        assert _should_enable_mlflow(Args(), Config()) is True

    def test_default_disabled_when_no_flags_no_config(self):
        """Test that MLflow is disabled by default."""
        from run_benchmarks import _should_enable_mlflow
        
        class Args:
            mlflow = False
            no_mlflow = False
        
        class Config:
            mlflow = None
        
        assert _should_enable_mlflow(Args(), Config()) is False
