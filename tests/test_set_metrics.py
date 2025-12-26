"""Tests for set-based metrics."""
import pytest
from metrics.set_metrics import (
    _parse_items,
    _parse_reference,
    compute_set_exact_match,
    compute_set_precision,
    compute_set_recall,
    compute_set_f1,
)


class TestParseItems:
    """Tests for _parse_items helper function."""
    
    def test_basic_newline_parsing(self):
        text = "item1\nitem2"
        result = _parse_items(text)
        assert result == {"item1", "item2"}
    
    def test_bracket_removal(self):
        text = "[123]\n[456]"
        result = _parse_items(text)
        assert result == {"123", "456"}
    
    def test_whitespace_handling(self):
        text = "  [123]  \n  [456]  "
        result = _parse_items(text)
        assert result == {"123", "456"}
    
    def test_empty_lines_filtered(self):
        text = "item1\n\nitem2"
        result = _parse_items(text)
        assert result == {"item1", "item2"}
    
    def test_empty_input(self):
        result = _parse_items("")
        assert result == set()
    
    def test_mixed_brackets_and_plain(self):
        text = "[123]\nplain_item\n[456]"
        result = _parse_items(text)
        assert result == {"123", "plain_item", "456"}
    
    def test_comma_separated(self):
        text = "item1, item2, item3"
        result = _parse_items(text)
        assert result == {"item1", "item2", "item3"}
    
    def test_comma_separated_with_brackets(self):
        text = "[123], [456], [789]"
        result = _parse_items(text)
        assert result == {"123", "456", "789"}
    
    def test_mixed_newlines_and_commas(self):
        text = "item1, item2\nitem3, item4"
        result = _parse_items(text)
        assert result == {"item1", "item2", "item3", "item4"}
    
    def test_quoted_items(self):
        text = '"item1", "item2"'
        result = _parse_items(text)
        assert result == {"item1", "item2"}


class TestParseReference:
    """Tests for _parse_reference helper function."""
    
    def test_basic_list(self):
        ref = ["a", "b", "c"]
        result = _parse_reference(ref)
        assert result == {"a", "b", "c"}
    
    def test_bracket_removal(self):
        ref = ["[123]", "[456]"]
        result = _parse_reference(ref)
        assert result == {"123", "456"}
    
    def test_whitespace_handling(self):
        ref = ["  [123]  ", "  456  "]
        result = _parse_reference(ref)
        assert result == {"123", "456"}
    
    def test_empty_list(self):
        result = _parse_reference([])
        assert result == set()


class TestSetExactMatch:
    """Tests for compute_set_exact_match."""
    
    def test_perfect_match(self):
        predictions = ["a\nb"]
        references = [["a", "b"]]
        result = compute_set_exact_match(predictions, references)
        assert result["set_exact_match"] == 1.0
    
    def test_partial_match_returns_zero(self):
        predictions = ["a\nb"]
        references = [["a", "c"]]
        result = compute_set_exact_match(predictions, references)
        assert result["set_exact_match"] == 0.0
    
    def test_order_independent(self):
        predictions = ["b\na"]
        references = [["a", "b"]]
        result = compute_set_exact_match(predictions, references)
        assert result["set_exact_match"] == 1.0
    
    def test_extra_items_returns_zero(self):
        predictions = ["a\nb\nc"]
        references = [["a", "b"]]
        result = compute_set_exact_match(predictions, references)
        assert result["set_exact_match"] == 0.0
    
    def test_multiple_examples(self):
        predictions = ["a\nb", "x\ny"]
        references = [["a", "b"], ["x", "z"]]
        result = compute_set_exact_match(predictions, references)
        assert result["set_exact_match"] == 0.5
    
    def test_empty_predictions(self):
        result = compute_set_exact_match([], [])
        assert result["set_exact_match"] == 0.0


class TestSetPrecision:
    """Tests for compute_set_precision."""
    
    def test_perfect_precision(self):
        predictions = ["a\nb"]
        references = [["a", "b", "c"]]
        result = compute_set_precision(predictions, references)
        assert result["set_precision"] == 1.0
    
    def test_partial_precision(self):
        predictions = ["a\nx"]
        references = [["a", "b"]]
        result = compute_set_precision(predictions, references)
        assert result["set_precision"] == 0.5
    
    def test_zero_precision(self):
        predictions = ["x\ny"]
        references = [["a", "b"]]
        result = compute_set_precision(predictions, references)
        assert result["set_precision"] == 0.0
    
    def test_empty_prediction(self):
        predictions = [""]
        references = [["a", "b"]]
        result = compute_set_precision(predictions, references)
        assert result["set_precision"] == 0.0


class TestSetRecall:
    """Tests for compute_set_recall."""
    
    def test_perfect_recall(self):
        predictions = ["a\nb\nc"]
        references = [["a", "b"]]
        result = compute_set_recall(predictions, references)
        assert result["set_recall"] == 1.0
    
    def test_partial_recall(self):
        predictions = ["a"]
        references = [["a", "b"]]
        result = compute_set_recall(predictions, references)
        assert result["set_recall"] == 0.5
    
    def test_zero_recall(self):
        predictions = ["x\ny"]
        references = [["a", "b"]]
        result = compute_set_recall(predictions, references)
        assert result["set_recall"] == 0.0
    
    def test_empty_reference(self):
        predictions = ["a\nb"]
        references = [[]]
        result = compute_set_recall(predictions, references)
        assert result["set_recall"] == 0.0


class TestSetF1:
    """Tests for compute_set_f1."""
    
    def test_perfect_f1(self):
        predictions = ["a\nb"]
        references = [["a", "b"]]
        result = compute_set_f1(predictions, references)
        assert result["set_f1"] == 1.0
    
    def test_balanced_partial(self):
        predictions = ["a\nx"]
        references = [["a", "b"]]
        result = compute_set_f1(predictions, references)
        # precision = 0.5, recall = 0.5, f1 = 0.5
        assert result["set_f1"] == 0.5
    
    def test_zero_f1(self):
        predictions = ["x"]
        references = [["a"]]
        result = compute_set_f1(predictions, references)
        assert result["set_f1"] == 0.0
    
    def test_high_precision_low_recall(self):
        predictions = ["a"]
        references = [["a", "b", "c", "d"]]
        result = compute_set_f1(predictions, references)
        # precision = 1.0, recall = 0.25, f1 = 2 * 1 * 0.25 / 1.25 = 0.4
        assert abs(result["set_f1"] - 0.4) < 0.001
    
    def test_multiple_examples(self):
        predictions = ["a\nb", "x"]
        references = [["a", "b"], ["x"]]
        result = compute_set_f1(predictions, references)
        # First: f1 = 1.0, Second: f1 = 1.0, avg = 1.0
        assert result["set_f1"] == 1.0


class TestWithBracketedIds:
    """Tests with bracketed IDs like ETHIC dataset."""
    
    def test_ethic_style_ids(self):
        predictions = ["[16356598]\n[15665713]"]
        references = [["[16356598]", "[15665713]"]]
        result = compute_set_exact_match(predictions, references)
        assert result["set_exact_match"] == 1.0
    
    def test_ethic_partial_match(self):
        predictions = ["[16356598]\n[99999999]"]
        references = [["[16356598]", "[15665713]"]]
        result = compute_set_precision(predictions, references)
        assert result["set_precision"] == 0.5
