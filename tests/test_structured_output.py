"""Unit tests for structured output parsing."""
import pytest
from models.structured_output import (
    MSMARCO_RESPONSE_SCHEMA,
    parse_json_response,
    parse_msmarco_response,
)


class TestParseJsonResponse:
    """Tests for parse_json_response function."""
    
    def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        text = '{"answer": "Paris", "passages": [0, 2]}'
        result = parse_json_response(text)
        
        assert result is not None
        assert result["answer"] == "Paris"
        assert result["passages"] == [0, 2]
    
    def test_parse_json_with_whitespace(self):
        """Test parsing JSON with leading/trailing whitespace."""
        text = '  \n{"answer": "London", "passages": [1]}  \n'
        result = parse_json_response(text)
        
        assert result is not None
        assert result["answer"] == "London"
    
    def test_parse_json_in_markdown_code_block(self):
        """Test extracting JSON from markdown code block."""
        text = '''Here is the answer:
```json
{"answer": "Tokyo", "passages": [3, 4]}
```
'''
        result = parse_json_response(text)
        
        assert result is not None
        assert result["answer"] == "Tokyo"
        assert result["passages"] == [3, 4]
    
    def test_parse_json_in_plain_code_block(self):
        """Test extracting JSON from plain code block."""
        text = '''```
{"answer": "Berlin", "passages": [0]}
```'''
        result = parse_json_response(text)
        
        assert result is not None
        assert result["answer"] == "Berlin"
    
    def test_parse_json_embedded_in_text(self):
        """Test extracting JSON embedded in text."""
        text = 'The answer is {"answer": "Madrid", "passages": [2]} based on the passages.'
        result = parse_json_response(text)
        
        assert result is not None
        assert result["answer"] == "Madrid"
    
    def test_parse_invalid_json_returns_none(self):
        """Test that invalid JSON returns None."""
        text = 'This is not JSON at all'
        result = parse_json_response(text)
        
        assert result is None
    
    def test_parse_malformed_json_returns_none(self):
        """Test that malformed JSON returns None."""
        text = '{"answer": "Paris", passages: [0]}'  # Missing quotes on key
        result = parse_json_response(text)
        
        assert result is None


class TestParseMsmarcoResponse:
    """Tests for parse_msmarco_response function."""
    
    def test_parse_valid_response(self):
        """Test parsing valid MS-MARCO response."""
        text = '{"answer": "42", "passages": [1, 3]}'
        result = parse_msmarco_response(text)
        
        assert result["answer"] == "42"
        assert result["passages"] == [1, 3]
    
    def test_parse_empty_passages(self):
        """Test parsing response with empty passages array."""
        text = '{"answer": "None found", "passages": []}'
        result = parse_msmarco_response(text)
        
        assert result["answer"] == "None found"
        assert result["passages"] == []
    
    def test_regex_fallback_for_answer(self):
        """Test regex fallback extracts answer."""
        text = 'I found the answer: "answer": "Einstein" in passage 2'
        result = parse_msmarco_response(text)
        
        assert result["answer"] == "Einstein"
    
    def test_regex_fallback_for_passages(self):
        """Test regex fallback extracts passages."""
        text = '"answer": "Yes", "passages": [0, 1, 2]'
        result = parse_msmarco_response(text)
        
        assert result["answer"] == "Yes"
        assert result["passages"] == [0, 1, 2]
    
    def test_completely_invalid_returns_nones(self):
        """Test that completely unparseable text returns None values."""
        text = 'Random text without any JSON structure'
        result = parse_msmarco_response(text)
        
        assert result["answer"] is None
        assert result["passages"] is None
    
    def test_validates_passages_as_integers(self):
        """Test that passage validation converts to integers."""
        text = '{"answer": "test", "passages": [0, 1, 2]}'
        result = parse_msmarco_response(text)
        
        assert all(isinstance(p, int) for p in result["passages"])


class TestMsmarcoSchema:
    """Tests for MSMARCO_RESPONSE_SCHEMA."""
    
    def test_schema_has_required_fields(self):
        """Test schema has correct structure."""
        assert "name" in MSMARCO_RESPONSE_SCHEMA
        assert "schema" in MSMARCO_RESPONSE_SCHEMA
        
        schema = MSMARCO_RESPONSE_SCHEMA["schema"]
        assert schema["type"] == "object"
        assert "answer" in schema["properties"]
        assert "passages" in schema["properties"]
        assert "answer" in schema["required"]
        assert "passages" in schema["required"]
    
    def test_schema_properties_types(self):
        """Test schema property types are correct."""
        schema = MSMARCO_RESPONSE_SCHEMA["schema"]
        
        assert schema["properties"]["answer"]["type"] == "string"
        assert schema["properties"]["passages"]["type"] == "array"
        assert schema["properties"]["passages"]["items"]["type"] == "integer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
