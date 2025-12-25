"""Tests for ETHIC dataset loader."""
import pytest
from unittest.mock import patch, MagicMock
from dataset.ethic import ETHICDataset, VALID_SUBSETS


class TestETHICDatasetInit:
    """Tests for ETHICDataset initialization."""
    
    def test_valid_subsets_constant(self):
        assert VALID_SUBSETS == ["Attributing", "Recalling", "Organizing"]
    
    def test_valid_subset_recalling(self):
        dataset = ETHICDataset(name="ethic", subset="Recalling")
        assert dataset.name == "ethic"
        assert dataset.subset == "Recalling"
    
    def test_valid_subset_attributing(self):
        dataset = ETHICDataset(name="ethic", subset="Attributing")
        assert dataset.subset == "Attributing"
    
    def test_valid_subset_organizing(self):
        dataset = ETHICDataset(name="ethic", subset="Organizing")
        assert dataset.subset == "Organizing"
    
    def test_invalid_subset_raises_error(self):
        with pytest.raises(ValueError) as excinfo:
            ETHICDataset(name="ethic", subset="InvalidSubset")
        assert "Invalid subset" in str(excinfo.value)
        assert "InvalidSubset" in str(excinfo.value)
    
    def test_summarizing_not_supported(self):
        with pytest.raises(ValueError) as excinfo:
            ETHICDataset(name="ethic", subset="Summarizing")
        assert "Invalid subset" in str(excinfo.value)
    
    def test_default_name(self):
        dataset = ETHICDataset(subset="Recalling")
        assert dataset.name == "ethic"


class TestETHICDatasetLoadExamples:
    """Tests for ETHICDataset.load_examples method."""
    
    @patch("dataset.ethic.hf_datasets.load_dataset")
    def test_load_examples_yields_correct_fields(self, mock_load):
        mock_data = [
            {
                "ID": "test_id_1",
                "Domain": "Medicine",
                "Task": "Citation",
                "Context": "Some context text",
                "System_msg": "System instructions",
                "User_msg": "User message with context",
                "Answer": ["[123]", "[456]"],
            }
        ]
        mock_load.return_value = mock_data
        
        dataset = ETHICDataset(name="ethic", subset="Recalling")
        examples = list(dataset.load_examples(split="test", limit=1))
        
        assert len(examples) == 1
        example = examples[0]
        
        # Check standard fields
        assert example["id"] == "test_id_1"
        assert example["context"] == "User message with context"
        assert example["question"] == ""
        assert example["answer"] == ["[123]", "[456]"]
        
        # Check metadata fields
        assert example["domain"] == "Medicine"
        assert example["ethic_task"] == "Citation"
        assert example["system_msg"] == "System instructions"
    
    @patch("dataset.ethic.hf_datasets.load_dataset")
    def test_load_examples_with_limit(self, mock_load):
        mock_data = [
            {"ID": f"id_{i}", "Domain": "D", "Task": "T", "Context": "C", 
             "System_msg": "S", "User_msg": "U", "Answer": []}
            for i in range(100)
        ]
        mock_load.return_value = mock_data
        
        dataset = ETHICDataset(name="ethic", subset="Recalling")
        examples = list(dataset.load_examples(split="test", limit=5, seed=42))
        
        assert len(examples) == 5
    
    @patch("dataset.ethic.hf_datasets.load_dataset")
    def test_deterministic_sampling(self, mock_load):
        mock_data = [
            {"ID": f"id_{i}", "Domain": "D", "Task": "T", "Context": "C",
             "System_msg": "S", "User_msg": "U", "Answer": []}
            for i in range(100)
        ]
        mock_load.return_value = mock_data
        
        dataset = ETHICDataset(name="ethic", subset="Recalling")
        
        # Same seed should produce same examples
        examples_1 = list(dataset.load_examples(split="test", limit=5, seed=42))
        examples_2 = list(dataset.load_examples(split="test", limit=5, seed=42))
        
        ids_1 = [e["id"] for e in examples_1]
        ids_2 = [e["id"] for e in examples_2]
        
        assert ids_1 == ids_2
    
    @patch("dataset.ethic.hf_datasets.load_dataset")
    def test_different_seeds_produce_different_samples(self, mock_load):
        mock_data = [
            {"ID": f"id_{i}", "Domain": "D", "Task": "T", "Context": "C",
             "System_msg": "S", "User_msg": "U", "Answer": []}
            for i in range(100)
        ]
        mock_load.return_value = mock_data
        
        dataset = ETHICDataset(name="ethic", subset="Recalling")
        
        examples_1 = list(dataset.load_examples(split="test", limit=5, seed=42))
        examples_2 = list(dataset.load_examples(split="test", limit=5, seed=123))
        
        ids_1 = [e["id"] for e in examples_1]
        ids_2 = [e["id"] for e in examples_2]
        
        assert ids_1 != ids_2
    
    @patch("dataset.ethic.hf_datasets.load_dataset")
    def test_load_calls_huggingface_correctly(self, mock_load):
        mock_load.return_value = []
        
        dataset = ETHICDataset(name="ethic", subset="Attributing")
        list(dataset.load_examples(split="test"))
        
        mock_load.assert_called_once_with("dmis-lab/ETHIC", "Attributing", split="test")


class TestETHICDatasetRepr:
    """Tests for ETHICDataset string representation."""
    
    def test_repr(self):
        dataset = ETHICDataset(name="ethic:Recalling", subset="Recalling")
        repr_str = repr(dataset)
        assert "ETHICDataset" in repr_str
        assert "ethic:Recalling" in repr_str
