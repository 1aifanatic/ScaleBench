"""Unit tests for retrieval task."""
import pytest
from tasks.retrieval_task import RetrievalTask
from models.structured_output import MSMARCO_RESPONSE_SCHEMA


class TestRetrievalTaskInit:
    """Tests for RetrievalTask initialization."""
    
    def test_default_init(self):
        """Test default initialization."""
        task = RetrievalTask()
        
        assert task.name == "retrieval_task"
        assert task.dataset_name is None
        assert task.response_schema == MSMARCO_RESPONSE_SCHEMA
    
    def test_init_with_dataset_name(self):
        """Test initialization with dataset name."""
        task = RetrievalTask(name="retrieval_task", dataset_name="msmarco")
        
        assert task.dataset_name == "msmarco"
    
    def test_loads_msmarco_prompt(self):
        """Test that msmarco dataset loads correct prompt."""
        task = RetrievalTask(dataset_name="msmarco")
        
        assert "JSON" in task.system_prompt
        assert "passages" in task.system_prompt.lower()


class TestBuildPrompt:
    """Tests for RetrievalTask.build_prompt."""
    
    def test_build_prompt_structure(self):
        """Test that build_prompt returns GenerationInput with correct fields."""
        task = RetrievalTask(dataset_name="msmarco")
        
        example = {
            "context": "[Passage 0]\nSome text.\n\n[Passage 1]\nMore text.",
            "question": "What is the capital?",
        }
        
        result = task.build_prompt(example)
        
        assert result.system_prompt is not None
        assert result.context is not None
        assert result.user_prompt is not None
        assert result.response_schema is not None
    
    def test_build_prompt_includes_schema(self):
        """Test that build_prompt includes response_schema."""
        task = RetrievalTask(dataset_name="msmarco")
        
        example = {
            "context": "Some context",
            "question": "A question?",
        }
        
        result = task.build_prompt(example)
        
        assert result.response_schema == MSMARCO_RESPONSE_SCHEMA
    
    def test_build_prompt_formats_context(self):
        """Test that context is prefixed with 'Passages:'."""
        task = RetrievalTask(dataset_name="msmarco")
        
        example = {
            "context": "[Passage 0]\nText here.",
            "question": "Question?",
        }
        
        result = task.build_prompt(example)
        
        assert result.context.startswith("Passages:")


class TestPostprocess:
    """Tests for RetrievalTask.postprocess."""
    
    def test_postprocess_valid_json(self):
        """Test postprocess with valid JSON."""
        task = RetrievalTask()
        
        output = '{"answer": "Paris", "passages": [0, 2]}'
        result = task.postprocess(output)
        
        assert isinstance(result, dict)
        assert result["answer"] == "Paris"
        assert result["passages"] == [0, 2]
    
    def test_postprocess_json_in_markdown(self):
        """Test postprocess extracts JSON from markdown."""
        task = RetrievalTask()
        
        output = '''Here's my answer:
```json
{"answer": "London", "passages": [1]}
```
'''
        result = task.postprocess(output)
        
        assert result["answer"] == "London"
        assert result["passages"] == [1]
    
    def test_postprocess_invalid_json_fallback(self):
        """Test postprocess falls back to regex for invalid JSON."""
        task = RetrievalTask()
        
        output = 'The "answer": "Berlin" is found in "passages": [0, 1]'
        result = task.postprocess(output)
        
        assert result["answer"] == "Berlin"
        assert result["passages"] == [0, 1]
    
    def test_postprocess_completely_invalid(self):
        """Test postprocess returns None values for unparseable output."""
        task = RetrievalTask()
        
        output = 'Random text without structure'
        result = task.postprocess(output)
        
        assert result["answer"] is None
        assert result["passages"] is None
    
    def test_postprocess_empty_passages(self):
        """Test postprocess handles empty passages array."""
        task = RetrievalTask()
        
        output = '{"answer": "Unknown", "passages": []}'
        result = task.postprocess(output)
        
        assert result["answer"] == "Unknown"
        assert result["passages"] == []


class TestNormalize:
    """Tests for RetrievalTask.normalize."""
    
    def test_normalize_lowercase(self):
        """Test normalization lowercases text."""
        task = RetrievalTask()
        
        result = task.normalize("PARIS")
        
        assert result == "paris"
    
    def test_normalize_removes_articles(self):
        """Test normalization removes articles."""
        task = RetrievalTask()
        
        result = task.normalize("the capital of France")
        
        assert "the" not in result
    
    def test_normalize_removes_punctuation(self):
        """Test normalization removes punctuation."""
        task = RetrievalTask()
        
        result = task.normalize("Paris, France!")
        
        assert "," not in result
        assert "!" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
