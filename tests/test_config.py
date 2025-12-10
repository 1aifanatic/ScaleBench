"""Tests for configuration loading and validation."""
import tempfile
from pathlib import Path
import pytest
import yaml

from core.config import (
    load_config,
    BenchmarkConfig,
    ExperimentConfig,
    ModelConfig,
    DatasetConfig,
    TaskConfig,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_valid_model_config(self):
        """Test creating a valid model config."""
        model = ModelConfig(name="openai:gpt-4", mode="baseline")
        assert model.name == "openai:gpt-4"
        assert model.mode == "baseline"

    def test_model_config_default_mode(self):
        """Test that mode defaults to 'baseline'."""
        model = ModelConfig(name="gemini:2.0-flash")
        assert model.mode == "baseline"

    def test_model_config_missing_name(self):
        """Test that name is required."""
        with pytest.raises(ValueError, match="model.name is required"):
            ModelConfig(name="")


class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_valid_dataset_config(self):
        """Test creating a valid dataset config."""
        dataset = DatasetConfig(
            name="longbench:narrativeqa",
            split="test",
            num_examples=100,
            seed=42,
        )
        assert dataset.name == "longbench:narrativeqa"
        assert dataset.split == "test"
        assert dataset.num_examples == 100
        assert dataset.seed == 42

    def test_dataset_config_defaults(self):
        """Test dataset config default values."""
        dataset = DatasetConfig(name="hotpotqa")
        assert dataset.split == "test"
        assert dataset.num_examples is None
        assert dataset.seed == 42

    def test_dataset_config_missing_name(self):
        """Test that name is required."""
        with pytest.raises(ValueError, match="dataset.name is required"):
            DatasetConfig(name="")


class TestTaskConfig:
    """Tests for TaskConfig dataclass."""

    def test_valid_task_config(self):
        """Test creating a valid task config."""
        task = TaskConfig(name="longdoc_qa")
        assert task.name == "longdoc_qa"

    def test_task_config_missing_name(self):
        """Test that name is required."""
        with pytest.raises(ValueError, match="task.name is required"):
            TaskConfig(name="")


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_valid_experiment_config(self):
        """Test creating a valid experiment config."""
        exp = ExperimentConfig(
            id="test_exp_1",
            model=ModelConfig(name="openai:gpt-4"),
            dataset=DatasetConfig(name="longbench:narrativeqa"),
            task=TaskConfig(name="longdoc_qa"),
            metrics=["f1", "rouge_l"],
        )
        assert exp.id == "test_exp_1"
        assert exp.model.name == "openai:gpt-4"
        assert exp.metrics == ["f1", "rouge_l"]

    def test_experiment_config_with_dict_nested_objects(self):
        """Test that nested dicts are converted to dataclass instances."""
        exp = ExperimentConfig(
            id="test_exp_2",
            model={"name": "gemini:2.0-flash", "mode": "baseline"},
            dataset={"name": "hotpotqa", "split": "test", "num_examples": 50},
            task={"name": "rag_task"},
        )
        assert isinstance(exp.model, ModelConfig)
        assert exp.model.name == "gemini:2.0-flash"
        assert isinstance(exp.dataset, DatasetConfig)
        assert exp.dataset.num_examples == 50

    def test_experiment_config_missing_id(self):
        """Test that id is required."""
        with pytest.raises(ValueError, match="experiment.id is required"):
            ExperimentConfig(
                id="",
                model=ModelConfig(name="openai:gpt-4"),
                dataset=DatasetConfig(name="longbench:narrativeqa"),
                task=TaskConfig(name="longdoc_qa"),
            )


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_valid_benchmark_config(self):
        """Test creating a valid benchmark config."""
        exp1 = ExperimentConfig(
            id="exp1",
            model=ModelConfig(name="openai:gpt-4"),
            dataset=DatasetConfig(name="longbench:narrativeqa"),
            task=TaskConfig(name="longdoc_qa"),
        )
        config = BenchmarkConfig(experiments=[exp1])
        assert len(config.experiments) == 1
        assert config.experiments[0].id == "exp1"

    def test_benchmark_config_empty_experiments(self):
        """Test that at least one experiment is required."""
        with pytest.raises(ValueError, match="At least one experiment is required"):
            BenchmarkConfig(experiments=[])

    def test_benchmark_config_with_dict_experiments(self):
        """Test that dict experiments are converted to ExperimentConfig."""
        config = BenchmarkConfig(
            experiments=[
                {
                    "id": "exp1",
                    "model": {"name": "openai:gpt-4"},
                    "dataset": {"name": "longbench:narrativeqa"},
                    "task": {"name": "longdoc_qa"},
                }
            ]
        )
        assert len(config.experiments) == 1
        assert isinstance(config.experiments[0], ExperimentConfig)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self):
        """Test loading a valid config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_data = {
                "experiments": [
                    {
                        "id": "test_exp",
                        "model": {"name": "openai:gpt-4"},
                        "dataset": {"name": "longbench:narrativeqa"},
                        "task": {"name": "longdoc_qa"},
                    }
                ]
            }
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            config = load_config(config_path)
            assert isinstance(config, BenchmarkConfig)
            assert len(config.experiments) == 1
            assert config.experiments[0].id == "test_exp"

    def test_load_config_file_not_found(self):
        """Test error when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.yaml"))

    def test_load_config_empty_file(self):
        """Test error when config file is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.touch()  # Create empty file

            with pytest.raises(ValueError, match="Config file is empty"):
                load_config(config_path)

    def test_load_config_missing_experiments_key(self):
        """Test error when 'experiments' key is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_data = {"some_other_key": []}
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            with pytest.raises(ValueError, match="'experiments' key is required"):
                load_config(config_path)

    def test_load_config_multiple_experiments(self):
        """Test loading config with multiple experiments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_data = {
                "experiments": [
                    {
                        "id": "exp1",
                        "model": {"name": "openai:gpt-4"},
                        "dataset": {"name": "longbench:narrativeqa"},
                        "task": {"name": "longdoc_qa"},
                        "metrics": ["f1"],
                    },
                    {
                        "id": "exp2",
                        "model": {"name": "gemini:2.0-flash"},
                        "dataset": {"name": "hotpotqa"},
                        "task": {"name": "rag_task"},
                        "metrics": ["em", "f1"],
                    },
                ]
            }
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            config = load_config(config_path)
            assert len(config.experiments) == 2
            assert config.experiments[0].id == "exp1"
            assert config.experiments[1].id == "exp2"
            assert config.experiments[1].metrics == ["em", "f1"]
