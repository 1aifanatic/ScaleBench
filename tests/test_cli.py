"""Tests for run_benchmarks CLI."""
import tempfile
from pathlib import Path
from unittest import mock
import yaml
import pytest

from run_benchmarks import main


@pytest.fixture
def temp_config():
    """Create a temporary valid config file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_data = {
            "experiments": [
                {
                    "id": "test_exp_1",
                    "model": {"name": "openai:gpt-4", "mode": "baseline"},
                    "dataset": {"name": "longbench:narrativeqa", "num_examples": 5},
                    "task": {"name": "longdoc_qa"},
                    "metrics": ["f1"],
                },
                {
                    "id": "test_exp_2",
                    "model": {"name": "gemini:2.0-flash", "mode": "baseline"},
                    "dataset": {"name": "hotpotqa", "num_examples": 3},
                    "task": {"name": "rag_task"},
                    "metrics": ["em", "f1"],
                },
            ]
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        yield config_path


class TestRunBenchmarksCLI:
    """Tests for run_benchmarks CLI."""

    def test_dry_run_with_default_config(self):
        """Test dry-run with the default config file."""
        with mock.patch("sys.argv", ["run_benchmarks.py", "--dry-run"]):
            # Should not raise an exception
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_dry_run_with_custom_config(self, temp_config):
        """Test dry-run with a custom config file."""
        with mock.patch(
            "sys.argv",
            ["run_benchmarks.py", "--config", str(temp_config), "--dry-run"],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_config_not_found(self, caplog):
        """Test error handling when config file doesn't exist."""
        with mock.patch(
            "sys.argv",
            ["run_benchmarks.py", "--config", "/nonexistent/config.yaml", "--dry-run"],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
            assert "Config file not found" in caplog.text or "Failed to load config" in caplog.text

    def test_experiment_id_filter_valid(self, temp_config):
        """Test filtering experiments by valid experiment ID."""
        with mock.patch(
            "sys.argv",
            [
                "run_benchmarks.py",
                "--config",
                str(temp_config),
                "--experiment-id",
                "test_exp_1",
                "--dry-run",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_experiment_id_filter_invalid(self, temp_config):
        """Test filtering experiments by invalid experiment ID."""
        with mock.patch(
            "sys.argv",
            [
                "run_benchmarks.py",
                "--config",
                str(temp_config),
                "--experiment-id",
                "nonexistent_exp",
                "--dry-run",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_verbose_flag(self, temp_config, capsys):
        """Test verbose logging flag."""
        with mock.patch(
            "sys.argv",
            ["run_benchmarks.py", "--config", str(temp_config), "--verbose", "--dry-run"],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_help_message(self, capsys):
        """Test that help message displays correctly."""
        with mock.patch("sys.argv", ["run_benchmarks.py", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # --help exits with code 0
            assert exc_info.value.code == 0
            captured = capsys.readouterr()
            assert "experiments.yaml" in captured.out
            assert "dry-run" in captured.out
