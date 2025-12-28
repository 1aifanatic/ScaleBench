"""Structured output schema definitions for model clients.

This module provides JSON Schema definitions for structured output responses.
The schemas are passed to GenerationInput.response_schema and handled natively
by each model client (OpenAI uses response_format, Gemini uses response_schema).

Usage:
    from models.structured_output import MSMARCO_RESPONSE_SCHEMA
    from models.base import GenerationInput
    
    inp = GenerationInput(
        user_prompt="...",
        response_schema=MSMARCO_RESPONSE_SCHEMA,
    )
    output = client.generate(inp)
    # output.text will be valid JSON matching the schema
"""
import json
import logging
import re
from typing import Dict, Any, Optional, List


logger = logging.getLogger(__name__)


# MS-MARCO response schema for answer + passage retrieval
# OpenAI requires: name, strict, schema with additionalProperties: false
# Gemini requires: just the schema object
MSMARCO_RESPONSE_SCHEMA: Dict[str, Any] = {
    "name": "msmarco_response",
    "schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The concise answer to the question"
            },
            "passages": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of 0-indexed passage numbers used to answer the question"
            }
        },
        "required": ["answer", "passages"],
        "additionalProperties": False
    }
}


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from model response text.
    
    Handles responses that may have markdown code blocks or extra text.
    
    Args:
        text: Raw response text from model.
        
    Returns:
        Parsed dict if successful, None otherwise.
    """
    text = text.strip()
    
    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try extracting JSON from markdown code blocks
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(code_block_pattern, text)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Try finding JSON object pattern in text
    json_pattern = r"\{[\s\S]*\}"
    matches = re.findall(json_pattern, text)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None


def parse_msmarco_response(text: str) -> Dict[str, Any]:
    """Parse MS-MARCO structured response with fallback.
    
    Args:
        text: Raw model response text (expected JSON).
        
    Returns:
        Dict with 'answer' (str or None) and 'passages' (List[int] or None).
    """
    # Try JSON parsing first
    parsed = parse_json_response(text)
    if parsed is not None:
        return {
            "answer": parsed.get("answer"),
            "passages": _validate_passages(parsed.get("passages")),
        }
    
    # Fallback to regex extraction
    logger.warning("JSON parsing failed, attempting regex fallback")
    return _regex_fallback_msmarco(text)


def _validate_passages(passages: Any) -> Optional[List[int]]:
    """Validate and convert passages to list of integers."""
    if passages is None:
        return None
    if not isinstance(passages, list):
        return None
    try:
        return [int(p) for p in passages]
    except (ValueError, TypeError):
        return None


def _regex_fallback_msmarco(text: str) -> Dict[str, Any]:
    """Extract answer and passages using regex as fallback."""
    result = {"answer": None, "passages": None}
    
    # Extract answer
    answer_patterns = [
        r'"answer"\s*:\s*"([^"]*)"',
        r'"answer"\s*:\s*\'([^\']*)\'',
        r'answer[:\s]+(.+?)(?:\n|$|,\s*["\']?passages)',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["answer"] = match.group(1).strip()
            break
    
    # Extract passages
    passages_patterns = [
        r'"passages"\s*:\s*\[([^\]]*)\]',
        r'passages[:\s]+\[([^\]]*)\]',
    ]
    for pattern in passages_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            passages_str = match.group(1)
            try:
                passages = [
                    int(p.strip()) 
                    for p in passages_str.split(",") 
                    if p.strip().lstrip("-").isdigit()
                ]
                result["passages"] = passages if passages else None
            except ValueError:
                pass
            break
    
    return result
