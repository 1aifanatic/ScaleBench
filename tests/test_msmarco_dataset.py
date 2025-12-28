"""Unit tests for MS-MARCO dataset loader.

Tests are organized into:
- Unit tests: Test internal logic with mocked data (fast, no network)
- Integration tests: Test actual data loading (requires network, marked with @pytest.mark.integration)
"""
import pytest
from dataset.msmarco import MSMARCODataset


# ============================================================================
# Unit Tests - Test internal logic with mocked data
# ============================================================================

class TestMSMARCODatasetInit:
    """Unit tests for dataset initialization."""

    def test_init_default_name(self):
        """Test MSMARCODataset initialization with default name."""
        dataset = MSMARCODataset(name="msmarco")
        assert dataset.name == "msmarco"

    def test_init_custom_name(self):
        """Test MSMARCODataset initialization with custom name."""
        dataset = MSMARCODataset(name="custom_msmarco")
        assert dataset.name == "custom_msmarco"


class TestBuildIndexedContext:
    """Unit tests for passage formatting logic."""

    def test_format_passages_basic(self):
        """Test passage formatting with multiple passages."""
        dataset = MSMARCODataset(name="msmarco")
        passages = ["First passage.", "Second passage.", "Third passage."]
        result = dataset._build_indexed_context(passages)
        
        assert "[Passage 0]" in result
        assert "[Passage 1]" in result
        assert "[Passage 2]" in result
        assert "First passage." in result
        assert "Second passage." in result
        assert "Third passage." in result

    def test_format_passages_single(self):
        """Test passage formatting with single passage."""
        dataset = MSMARCODataset(name="msmarco")
        passages = ["Only passage."]
        result = dataset._build_indexed_context(passages)
        
        assert "[Passage 0]" in result
        assert "Only passage." in result
        assert "[Passage 1]" not in result

    def test_format_passages_empty(self):
        """Test passage formatting with empty list."""
        dataset = MSMARCODataset(name="msmarco")
        passages = []
        result = dataset._build_indexed_context(passages)
        
        assert result == ""

    def test_format_passages_preserves_content(self):
        """Test that passage content is preserved exactly."""
        dataset = MSMARCODataset(name="msmarco")
        passages = ["Content with special chars: @#$%"]
        result = dataset._build_indexed_context(passages)
        
        assert "Content with special chars: @#$%" in result


class TestHasValidAnswer:
    """Unit tests for answer validation logic."""

    def test_valid_answer(self):
        """Test with a valid answer."""
        dataset = MSMARCODataset(name="msmarco")
        assert dataset._has_valid_answer(["This is a valid answer."])

    def test_no_answer_present_only(self):
        """Test with only 'No Answer Present.'."""
        dataset = MSMARCODataset(name="msmarco")
        assert not dataset._has_valid_answer(["No Answer Present."])

    def test_empty_answers(self):
        """Test with empty answer list."""
        dataset = MSMARCODataset(name="msmarco")
        assert not dataset._has_valid_answer([])

    def test_mixed_answers(self):
        """Test with mixed valid and invalid answers."""
        dataset = MSMARCODataset(name="msmarco")
        answers = ["No Answer Present.", "This is valid."]
        assert dataset._has_valid_answer(answers)


class TestHasRetrievalGroundTruth:
    """Unit tests for retrieval ground truth validation."""

    def test_has_selected_passage(self):
        """Test with at least one selected passage."""
        dataset = MSMARCODataset(name="msmarco")
        assert dataset._has_retrieval_ground_truth([0, 1, 0])

    def test_no_selected_passages(self):
        """Test with no selected passages."""
        dataset = MSMARCODataset(name="msmarco")
        assert not dataset._has_retrieval_ground_truth([0, 0, 0])

    def test_all_selected(self):
        """Test with all passages selected."""
        dataset = MSMARCODataset(name="msmarco")
        assert dataset._has_retrieval_ground_truth([1, 1, 1])


class TestSelectedIndices:
    """Unit tests for selected indices extraction logic."""

    def test_extract_selected_indices_basic(self):
        """Test extracting selected indices from labels."""
        labels = [0, 1, 0, 1, 0]
        expected = [1, 3]
        
        result = [i for i, label in enumerate(labels) if label == 1]
        assert result == expected

    def test_extract_selected_indices_all_selected(self):
        """Test when all passages are selected."""
        labels = [1, 1, 1]
        expected = [0, 1, 2]
        
        result = [i for i, label in enumerate(labels) if label == 1]
        assert result == expected

    def test_extract_selected_indices_none_selected(self):
        """Test when no passages are selected."""
        labels = [0, 0, 0]
        expected = []
        
        result = [i for i, label in enumerate(labels) if label == 1]
        assert result == expected


# ============================================================================
# Integration Tests - Require network access to load actual data
# ============================================================================

@pytest.mark.integration
class TestMSMARCOIntegration:
    """Integration tests that load actual MS-MARCO data from HuggingFace."""

    def test_load_examples_structure(self):
        """Test MS-MARCO examples have correct structure."""
        dataset = MSMARCODataset(name="msmarco")
        examples = list(dataset.load_examples(split="validation", limit=1, seed=42))
        
        assert len(examples) == 1
        ex = examples[0]
        
        # Check required fields
        assert "id" in ex
        assert "context" in ex
        assert "question" in ex
        assert "answer" in ex
        assert "well_formed_answers" in ex
        assert "passage_labels" in ex
        assert "selected_indices" in ex
        
        # Check types
        assert isinstance(ex["id"], str)
        assert isinstance(ex["context"], str)
        assert isinstance(ex["question"], str)
        assert isinstance(ex["answer"], list)
        assert isinstance(ex["well_formed_answers"], list)
        assert isinstance(ex["passage_labels"], list)
        assert isinstance(ex["selected_indices"], list)

    def test_context_has_passage_prefixes(self):
        """Test that context contains [Passage N] prefixes."""
        dataset = MSMARCODataset(name="msmarco")
        examples = list(dataset.load_examples(split="validation", limit=1, seed=42))
        
        ex = examples[0]
        context = ex["context"]
        
        assert "[Passage 0]" in context

    def test_selected_indices_match_labels(self):
        """Test that selected_indices matches passage_labels."""
        dataset = MSMARCODataset(name="msmarco")
        examples = list(dataset.load_examples(split="validation", limit=5, seed=42))
        
        for ex in examples:
            labels = ex["passage_labels"]
            indices = ex["selected_indices"]
            
            expected_indices = [i for i, label in enumerate(labels) if label == 1]
            assert indices == expected_indices

    def test_filters_no_answer_present(self):
        """Test that examples with 'No Answer Present.' are filtered out."""
        dataset = MSMARCODataset(name="msmarco")
        examples = list(dataset.load_examples(split="validation", limit=20, seed=42))
        
        for ex in examples:
            answers = ex["answer"]
            assert not (len(answers) == 1 and answers[0] == "No Answer Present.")

    def test_has_retrieval_ground_truth(self):
        """Test that all examples have at least one selected passage."""
        dataset = MSMARCODataset(name="msmarco")
        examples = list(dataset.load_examples(split="validation", limit=20, seed=42))
        
        for ex in examples:
            indices = ex["selected_indices"]
            assert len(indices) >= 1

    def test_deterministic_sampling(self):
        """Test that same seed produces same examples."""
        dataset = MSMARCODataset(name="msmarco")
        
        examples1 = list(dataset.load_examples(split="validation", limit=3, seed=42))
        examples2 = list(dataset.load_examples(split="validation", limit=3, seed=42))
        
        assert len(examples1) == 3
        assert len(examples2) == 3
        
        ids1 = [ex["id"] for ex in examples1]
        ids2 = [ex["id"] for ex in examples2]
        assert ids1 == ids2

    def test_different_seeds_different_examples(self):
        """Test that different seeds produce different examples."""
        dataset = MSMARCODataset(name="msmarco")
        
        examples1 = list(dataset.load_examples(split="validation", limit=3, seed=42))
        examples2 = list(dataset.load_examples(split="validation", limit=3, seed=123))
        
        ids1 = [ex["id"] for ex in examples1]
        ids2 = [ex["id"] for ex in examples2]
        
        assert ids1 != ids2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
