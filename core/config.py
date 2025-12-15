"""Configuration management: environment setup and experiment YAML parsing."""
import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    mode: str = "baseline"  # 'baseline' or 'scaledown'
    scaledown: Optional[dict] = None  # ScaleDown config (rate, etc.)
    generation: Optional[dict] = None  # Optional generation params (temperature, top_p, max_tokens)
    pricing: Optional[dict] = None  # Optional pricing override (input_per_1m_tokens, output_per_1m_tokens)
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("model.name is required")


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    split: str = "test"
    num_examples: Optional[int] = None
    seed: int = 42
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("dataset.name is required")


@dataclass
class TaskConfig:
    """Task configuration."""
    name: str
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("task.name is required")


@dataclass
class ExperimentConfig:
    """Single experiment configuration."""
    id: str
    model: ModelConfig
    dataset: DatasetConfig
    task: TaskConfig
    metrics: Optional[List[str]] = None
    
    def __post_init__(self):
        if not self.id:
            raise ValueError("experiment.id is required")
        if not isinstance(self.model, ModelConfig):
            self.model = ModelConfig(**self.model) if isinstance(self.model, dict) else self.model
        if not isinstance(self.dataset, DatasetConfig):
            self.dataset = DatasetConfig(**self.dataset) if isinstance(self.dataset, dict) else self.dataset
        if not isinstance(self.task, TaskConfig):
            self.task = TaskConfig(**self.task) if isinstance(self.task, dict) else self.task


@dataclass
class BenchmarkConfig:
    """Root benchmark configuration."""
    experiments: List[ExperimentConfig]
    
    def __post_init__(self):
        if not self.experiments:
            raise ValueError("At least one experiment is required")
        # Ensure all experiments are ExperimentConfig instances
        validated_exps = []
        for exp in self.experiments:
            if isinstance(exp, dict):
                validated_exps.append(ExperimentConfig(
                    id=exp.get("id", ""),
                    model=ModelConfig(**exp.get("model", {})),
                    dataset=DatasetConfig(**exp.get("dataset", {})),
                    task=TaskConfig(**exp.get("task", {})),
                    metrics=exp.get("metrics"),
                ))
            else:
                validated_exps.append(exp)
        self.experiments = validated_exps


def load_config(config_path: Path) -> BenchmarkConfig:
    """Load and validate benchmark configuration from YAML.
    
    Args:
        config_path: Path to experiments.yaml file.
        
    Returns:
        BenchmarkConfig object with validated experiments.
        
    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If config is invalid.
        yaml.YAMLError: If YAML parsing fails.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    
    if not data:
        raise ValueError("Config file is empty")
    
    if "experiments" not in data:
        raise ValueError("'experiments' key is required in config")
    
    return BenchmarkConfig(experiments=data["experiments"])
