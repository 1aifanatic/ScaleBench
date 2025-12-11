"""Unit tests for dataset loaders."""
import pytest
from dataset.base import Example, Dataset
from dataset.hotpotqa import HotpotQADataset


def test_example_dict():
    """Test Example dict subclass."""
    ex = Example(id="1", context="ctx", question="q?", answer=["a"])
    assert ex["id"] == "1"
    assert ex["context"] == "ctx"
    assert ex["question"] == "q?"
    assert ex["answer"] == ["a"]


def test_hotpotqa_dataset_init():
    """Test HotpotQADataset initialization."""
    dataset = HotpotQADataset(name="hotpotqa")
    assert dataset.name == "hotpotqa"
    assert dataset.subset == "distractor"


def test_hotpotqa_dataset_with_subset():
    """Test HotpotQADataset with fullwiki subset."""
    dataset = HotpotQADataset(name="hotpotqa:fullwiki", subset="fullwiki")
    assert dataset.name == "hotpotqa:fullwiki"
    assert dataset.subset == "fullwiki"


def test_hotpotqa_load_examples_structure():
    """Test HotpotQA examples have correct structure."""
    dataset = HotpotQADataset(name="hotpotqa")
    
    # Load just one example
    examples = list(dataset.load_examples(split="validation", limit=1, seed=42))
    
    assert len(examples) == 1
    ex = examples[0]
    
    # Check required fields
    assert "id" in ex
    assert "context" in ex
    assert "question" in ex
    assert "answer" in ex
    
    # Check types
    assert isinstance(ex["id"], str)
    assert isinstance(ex["context"], str)
    assert isinstance(ex["question"], str)
    assert isinstance(ex["answer"], list)
    assert len(ex["answer"]) > 0
    assert isinstance(ex["answer"][0], str)


def test_hotpotqa_deterministic_sampling():
    """Test that same seed produces same examples."""
    dataset = HotpotQADataset(name="hotpotqa")
    
    examples1 = list(dataset.load_examples(split="validation", limit=3, seed=42))
    examples2 = list(dataset.load_examples(split="validation", limit=3, seed=42))
    
    assert len(examples1) == 3
    assert len(examples2) == 3
    
    # Same IDs in same order
    ids1 = [ex["id"] for ex in examples1]
    ids2 = [ex["id"] for ex in examples2]
    assert ids1 == ids2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
