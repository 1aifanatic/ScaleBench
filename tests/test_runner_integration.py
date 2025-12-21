"""Minimal integration test for runner and aggregation."""
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from core.config import ExperimentConfig, ModelConfig, DatasetConfig, TaskConfig
from evaluation.runner import run_experiment
from evaluation.aggregation import aggregate_from_jsonl, save_summary_csv, save_summary_json
from models.base import ModelOutput, GenerationInput
from dataset.base import Example


def test_runner_with_mock_components():
    """Test runner with mocked model client."""
    
    # Create mock dataset
    mock_dataset = Mock()
    mock_dataset.load_examples.return_value = iter([
        Example(id="ex1", context="Context 1", question="Q1?", answer=["answer1"]),
        Example(id="ex2", context="Context 2", question="Q2?", answer=["answer2"]),
        Example(id="ex3", context="Context 3", question="Q3?", answer=["answer3"]),
    ])
    
    # Create mock task
    mock_task = Mock()
    mock_task.build_prompt.side_effect = lambda ex, **kwargs: GenerationInput(
        user_prompt=f"Question: {ex['question']}",
        context=ex.get("context"),
        **kwargs
    )
    mock_task.postprocess.side_effect = lambda text: text.strip().lower()
    mock_task.normalize.side_effect = lambda text: text.strip().lower()
    
    # Create mock model client
    mock_model = Mock()
    mock_model.generate.side_effect = [
        ModelOutput(text="Answer1", latency_ms=100, input_tokens=10, output_tokens=5),
        ModelOutput(text="Answer2", latency_ms=120, input_tokens=12, output_tokens=6),
        ModelOutput(text="Answer3", latency_ms=110, input_tokens=11, output_tokens=5),
    ]
    
    # Create mock metrics (match actual function signature)
    mock_metrics = {
        "exact_match": lambda predictions, references, normalize_fn=None: {"exact_match": 1.0},
        "f1": lambda predictions, references, normalize_fn=None: {"f1": 0.95},
    }
    
    # Create experiment config
    config = ExperimentConfig(
        id="test_exp",
        model=ModelConfig(name="test:model"),
        dataset=DatasetConfig(name="test", split="test", num_examples=3, seed=42),
        task=TaskConfig(name="test_task"),
        metrics=["exact_match", "f1"],
    )
    
    # Patch registries
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Temporarily change results directory
        with patch("evaluation.runner.Path") as mock_path_class:
            mock_raw_dir = tmpdir / "results" / "raw"
            mock_raw_dir.mkdir(parents=True, exist_ok=True)
            mock_path_class.return_value = mock_raw_dir
            
            # Patch get functions
            with patch("evaluation.runner.get_model_client", return_value=mock_model), \
                 patch("evaluation.runner.get_dataset", return_value=mock_dataset), \
                 patch("evaluation.runner.get_task", return_value=mock_task), \
                 patch("evaluation.runner.get_metrics", return_value=mock_metrics):
                
                # Run experiment
                result = run_experiment(config)
                
                # Verify result structure
                assert result["experiment_id"] == "test_exp"
                assert result["model"] == "test:model"
                assert result["n_examples"] == 3
                assert "exact_match" in result["metrics"]
                assert "f1" in result["metrics"]
                assert result["avg_latency_ms"] > 0
                assert result["avg_input_tokens"] > 0
                assert result["avg_output_tokens"] > 0
                
                print("✓ Runner test passed")
                print(f"  - n_examples: {result['n_examples']}")
                print(f"  - exact_match: {result['metrics']['exact_match']:.2f}")
                print(f"  - f1: {result['metrics']['f1']:.2f}")


def test_aggregation_from_jsonl():
    """Test aggregation functions with sample JSONL."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create sample JSONL file
        jsonl_file = tmpdir / "test_experiment.jsonl"
        
        sample_data = [
            {
                "id": "ex1",
                "model": "test:model",
                "pred": "answer1",
                "gold": ["answer1"],
                "input_tokens": 10,
                "output_tokens": 5,
                "latency_ms": 100.5,
                "metrics": {"exact_match": 1.0, "f1": 0.95},
            },
            {
                "id": "ex2",
                "model": "test:model",
                "pred": "answer2",
                "gold": ["answer2"],
                "input_tokens": 12,
                "output_tokens": 6,
                "latency_ms": 120.3,
                "metrics": {"exact_match": 0.0, "f1": 0.85},
            },
            {
                "id": "ex3",
                "model": "test:model",
                "pred": "answer3",
                "gold": ["answer3"],
                "input_tokens": 11,
                "output_tokens": 5,
                "latency_ms": 110.2,
                "metrics": {"exact_match": 1.0, "f1": 1.0},
            },
        ]
        
        with open(jsonl_file, "w") as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")
        
        # Test aggregation
        aggregated = aggregate_from_jsonl(jsonl_file)
        
        assert aggregated["experiment_id"] == "test_experiment"
        assert aggregated["model"] == "test:model"
        assert aggregated["n_examples"] == 3
        
        # Check metric stats
        assert "exact_match" in aggregated["metrics"]
        assert "f1" in aggregated["metrics"]
        assert abs(aggregated["metrics"]["exact_match"]["mean"] - 0.6667) < 0.01
        assert aggregated["metrics"]["f1"]["p50"] == 0.95
        
        # Check latency stats
        assert aggregated["latency_ms"]["mean"] > 100
        assert aggregated["latency_ms"]["p50"] > 0
        
        print("✓ Aggregation test passed")
        print(f"  - exact_match mean: {aggregated['metrics']['exact_match']['mean']:.2f}")
        print(f"  - f1 mean: {aggregated['metrics']['f1']['mean']:.2f}")
        print(f"  - latency p50: {aggregated['latency_ms']['p50']:.2f}ms")
        
        # Test CSV export
        csv_file = tmpdir / "summary.csv"
        save_summary_csv(aggregated, csv_file)
        assert csv_file.exists()
        print(f"✓ CSV export test passed: {csv_file}")
        
        # Test JSON export
        json_file = tmpdir / "summary.json"
        save_summary_json(aggregated, json_file)
        assert json_file.exists()
        print(f"✓ JSON export test passed: {json_file}")


if __name__ == "__main__":
    print("Running minimal integration tests...\n")
    test_runner_with_mock_components()
    print()
    test_aggregation_from_jsonl()
    print("\n✓✓✓ All integration tests passed!")


def test_longbench_single_task_experiment():
    """Test LongBench single-task experiment with mocked components."""
    
    # Create mock dataset
    mock_dataset = Mock()
    mock_dataset.load_examples.return_value = iter([
        Example(id="narrativeqa_1", context="Long context 1", question="Q1?", answer=["answer1"]),
        Example(id="narrativeqa_2", context="Long context 2", question="Q2?", answer=["answer2"]),
    ])
    
    # Create mock task
    mock_task = Mock()
    mock_task.build_prompt.side_effect = lambda ex, **kwargs: GenerationInput(
        user_prompt=f"Question: {ex['question']}",
        context=ex.get("context"),
        **kwargs
    )
    mock_task.postprocess.side_effect = lambda text: text.strip().lower()
    mock_task.normalize.side_effect = lambda text: text.strip().lower()
    
    # Create mock model client
    mock_model = Mock()
    mock_model.generate.side_effect = [
        ModelOutput(text="Answer1", latency_ms=100, input_tokens=50, output_tokens=5),
        ModelOutput(text="Answer2", latency_ms=120, input_tokens=55, output_tokens=6),
    ]
    
    # Create mock metrics
    mock_metrics = {
        "f1": lambda predictions, references, normalize_fn=None: {"f1": 0.90},
        "bertscore": lambda predictions, references, normalize_fn=None: {
            "bertscore_precision": 0.85,
            "bertscore_recall": 0.88,
            "bertscore_f1": 0.86,
        },
    }
    
    # Create experiment config for single task
    config = ExperimentConfig(
        id="longbench_narrativeqa_test",
        model=ModelConfig(name="test:model"),
        dataset=DatasetConfig(name="longbench", split="test", task="narrativeqa", num_examples=2, seed=42),
        task=TaskConfig(name="rag_qa"),
        metrics=["f1", "bertscore"],
    )
    
    # Patch registries
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        mock_raw_dir = tmpdir / "results" / "raw"
        mock_raw_dir.mkdir(parents=True, exist_ok=True)
        
        with patch("evaluation.runner.Path") as mock_path_class:
            mock_path_class.return_value = mock_raw_dir
            
            with patch("evaluation.runner.get_model_client", return_value=mock_model), \
                 patch("evaluation.runner.get_dataset", return_value=mock_dataset), \
                 patch("evaluation.runner.get_task", return_value=mock_task), \
                 patch("evaluation.runner.get_metrics", return_value=mock_metrics):
                
                # Run experiment
                result = run_experiment(config)
                
                # Verify result structure
                assert result["experiment_id"] == "longbench_narrativeqa_test"
                assert result["model"] == "test:model"
                assert result["dataset"] == "longbench"
                assert result["n_examples"] == 2
                assert "f1" in result["metrics"]
                assert "bertscore_precision" in result["metrics"]
                assert "bertscore_recall" in result["metrics"]
                assert "bertscore_f1" in result["metrics"]
                assert result["avg_latency_ms"] > 0
                assert result["avg_input_tokens"] > 0
                
                print("✓ LongBench single-task test passed")
                print(f"  - task: narrativeqa")
                print(f"  - n_examples: {result['n_examples']}")
                print(f"  - f1: {result['metrics']['f1']:.2f}")
                print(f"  - bertscore_f1: {result['metrics']['bertscore_f1']:.2f}")


def test_longbench_multi_task_experiment():
    """Test LongBench multi-task experiment with mocked components."""
    from dataclasses import replace
    from dataset.longbench import VALID_TASKS
    
    # Create experiment config for multi-task (task=None)
    config = ExperimentConfig(
        id="longbench_all_test",
        model=ModelConfig(name="test:model"),
        dataset=DatasetConfig(name="longbench", split="test", task=None, num_examples=2, seed=42),
        task=TaskConfig(name="rag_qa"),
        metrics=["f1"],
    )
    
    # Mock the single-task experiment function
    def mock_single_task_exp(task_config):
        task_name = task_config.dataset.task
        return {
            "experiment_id": f"{config.id}_{task_name}",
            "model": "test:model",
            "dataset": "longbench",
            "n_examples": 2,
            "metrics": {"f1": 0.85 + (0.05 * VALID_TASKS.index(task_name))},
            "avg_latency_ms": 100.0,
            "avg_input_tokens": 50.0,
            "avg_output_tokens": 5.0,
            "total_cost_usd": 0.01,
            "avg_cost_usd": 0.005,
            "raw_output_file": f"results/raw/{config.id}_{task_name}_20251218_120000.jsonl",
        }
    
    with patch("evaluation.runner._run_single_task_experiment", side_effect=mock_single_task_exp):
        # Import here to access the actual multi-task function
        from evaluation.runner import _run_multi_task_experiment
        
        result = _run_multi_task_experiment(config)
        
        # Verify multi-task result structure
        assert result["experiment_id"] == "longbench_all_test"
        assert result["is_multi_task"] is True
        assert result["n_examples"] == 8  # 2 examples * 4 tasks
        assert "per_task" in result
        assert len(result["per_task"]) == 4
        
        # Check all tasks present
        for task in VALID_TASKS:
            assert task in result["per_task"]
            assert result["per_task"][task]["n_examples"] == 2
        
        # Check aggregated metrics exist
        assert "f1" in result["metrics"]
        assert result["avg_latency_ms"] > 0
        
        print("✓ LongBench multi-task test passed")
        print(f"  - tasks: {len(result['per_task'])}")
        print(f"  - total n_examples: {result['n_examples']}")
        print(f"  - overall f1: {result['metrics']['f1']:.2f}")
        print(f"  - per_task breakdown:")
        for task, task_result in result["per_task"].items():
            print(f"    - {task}: f1={task_result['metrics']['f1']:.2f}")
