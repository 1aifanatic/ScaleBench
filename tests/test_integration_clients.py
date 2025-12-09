"""Integration tests for model clients with real APIs.

These tests require API keys to be set in environment variables.
They are separated from unit tests to allow running without API access.

Run with: pytest tests/test_integration_clients.py -v
Skip with markers: pytest tests/ -m "not integration"
"""
import os
import pytest

from models.base import GenerationInput, PricingInfo
from models.openai_client import OpenAIClient


# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def skip_if_no_openai_key():
    """Skip test if OpenAI API key is not available."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


class TestOpenAIClientIntegration:
    """Integration tests for OpenAI client with real API."""

    def test_simple_generation(self, skip_if_no_openai_key):
        """Test basic text generation with OpenAI."""
        client = OpenAIClient(model_name="gpt-4o-mini")
        
        inp = GenerationInput(user_prompt="Say 'Hello, World!' and nothing else.")
        output = client.generate(inp)
        
        assert output.text is not None
        assert len(output.text) > 0
        assert output.latency_ms > 0
        assert output.input_tokens > 0
        assert output.output_tokens > 0

    def test_with_system_prompt(self, skip_if_no_openai_key):
        """Test generation with system prompt."""
        client = OpenAIClient(model_name="gpt-4o-mini")
        
        inp = GenerationInput(
            system_prompt="You are a helpful assistant that only responds in JSON.",
            user_prompt='Return {"status": "ok"}',
        )
        output = client.generate(inp)
        
        assert output.text is not None
        assert "ok" in output.text.lower() or "status" in output.text.lower()

    def test_with_temperature(self, skip_if_no_openai_key):
        """Test generation with custom temperature."""
        client = OpenAIClient(model_name="gpt-4o-mini")
        
        inp = GenerationInput(
            user_prompt="Say 'test' and nothing else.",
            temperature=0.0,  # Deterministic
        )
        output = client.generate(inp)
        
        assert output.text is not None
        assert len(output.text) > 0

    def test_with_max_tokens(self, skip_if_no_openai_key):
        """Test generation with max output tokens limit."""
        client = OpenAIClient(model_name="gpt-4o-mini")
        
        inp = GenerationInput(
            user_prompt="Write a long story.",
            max_output_tokens=20,
        )
        output = client.generate(inp)
        
        assert output.text is not None
        assert output.output_tokens <= 25  # Allow small buffer

    def test_token_counting(self, skip_if_no_openai_key):
        """Test tiktoken token counting."""
        client = OpenAIClient(model_name="gpt-4o-mini")
        
        text = "Hello, world!"
        count = client.count_tokens(text)
        
        assert count > 0
        assert count < 10  # Should be a few tokens

    def test_with_pricing(self, skip_if_no_openai_key):
        """Test cost calculation with pricing configured."""
        pricing = PricingInfo(
            input_per_1k_tokens=0.00015,  # gpt-4o-mini pricing (example)
            output_per_1k_tokens=0.0006,
        )
        client = OpenAIClient(model_name="gpt-4o-mini", pricing=pricing)
        
        inp = GenerationInput(user_prompt="Say hello.")
        output = client.generate(inp)
        
        assert output.cost_usd is not None
        assert output.cost_usd > 0
        assert output.cost_usd < 0.01  # Should be tiny cost

    def test_with_long_context(self, skip_if_no_openai_key):
        """Test handling long context."""
        client = OpenAIClient(model_name="gpt-4o-mini")
        
        long_context = "Context: " + ("word " * 1000)
        inp = GenerationInput(
            context=long_context,
            user_prompt="What is the context about?",
        )
        output = client.generate(inp)
        
        assert output.text is not None
        assert output.input_tokens > 1000  # Should count context tokens
