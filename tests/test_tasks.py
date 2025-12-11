"""Unit tests for tasks."""
import pytest
from tasks.base import Task
from tasks.rag_task import RAGTask
from dataset.base import Example


def test_task_normalize():
    """Test Task.normalize() function."""
    task = RAGTask(name="test")
    
    # Test lowercase
    assert task.normalize("HELLO") == "hello"
    
    # Test article removal
    assert task.normalize("the cat") == "cat"
    assert task.normalize("a dog") == "dog"
    assert task.normalize("an apple") == "apple"
    
    # Test punctuation removal
    assert task.normalize("hello, world!") == "hello world"
    assert task.normalize("what?") == "what"
    
    # Test whitespace collapse
    assert task.normalize("hello   world") == "hello world"
    assert task.normalize("  hello  ") == "hello"
    
    # Test combined
    assert task.normalize("The cat, jumped!") == "cat jumped"


def test_rag_task_build_prompt():
    """Test RAGTask.build_prompt() function."""
    task = RAGTask(name="rag_qa")
    
    example = Example(
        id="1",
        context="Paris is the capital of France. France is in Europe.",
        question="What is the capital of France?",
        answer=["Paris"],
    )
    
    prompt = task.build_prompt(example)
    
    # Check prompt contains key elements
    assert "Paris is the capital of France" in prompt
    assert "What is the capital of France?" in prompt
    assert len(prompt) > 50  # Has instruction text


def test_rag_task_postprocess():
    """Test RAGTask.postprocess() function."""
    task = RAGTask(name="rag_qa")
    
    # Test single line
    assert task.postprocess("Paris") == "Paris"
    
    # Test multiline (takes first non-empty)
    output = "Paris\n\nExplanation: Paris is the capital..."
    assert task.postprocess(output) == "Paris"
    
    # Test with leading/trailing whitespace
    assert task.postprocess("  Paris  \n") == "Paris"
    
    # Test empty lines
    output = "\n\nParis\nOther text"
    assert task.postprocess(output) == "Paris"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
