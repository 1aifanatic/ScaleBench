"""Integration tests for model clients with real APIs.

These tests require API keys to be set in environment variables.
They are separated from unit tests to allow running without API access.

Run with: pytest tests/test_integration_clients.py -v
Run only ScaleDown tests: pytest tests/test_integration_clients.py::TestScaleDownClientIntegration -v
Skip with markers: pytest tests/ -m "not integration"
"""
import os
import logging
import pytest
import wikipedia

from models.base import GenerationInput, PricingInfo
from models.openai_client import OpenAIClient
from models.gemini_client import GeminiClient
from models.scaledown_client import ScaleDownClient


# Configure logging for test responses
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def skip_if_no_openai_key():
    """Skip test if OpenAI API key is not available."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


@pytest.fixture
def skip_if_no_gemini_key():
    """Skip test if Gemini API key is not available."""
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")


@pytest.fixture
def skip_if_no_scaledown_key():
    """Skip test if ScaleDown API key is not available."""
    if not os.getenv("SCALEDOWN_API_KEY"):
        pytest.skip("SCALEDOWN_API_KEY not set")


@pytest.fixture
def wikipedia_context():
    """Fetch Wikipedia content for testing."""
    try:
        page = wikipedia.page('Taylor Swift', auto_suggest=False)
        logger.info(f"Fetched Wikipedia page: {page.title}")
        logger.info(f"Content length: {len(page.content)} characters")
        return page.content
    except wikipedia.exceptions.DisambiguationError:
        pytest.skip("Wikipedia disambiguation page encountered")
    except wikipedia.exceptions.PageError:
        pytest.skip("Wikipedia page not found")


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


class TestGeminiClientIntegration:
    """Integration tests for Gemini client with real API."""

    def test_simple_generation(self, skip_if_no_gemini_key):
        """Test basic text generation with Gemini."""
        client = GeminiClient(model_name="gemini-2.5-flash")
        
        inp = GenerationInput(user_prompt="Say 'Hello, World!' and nothing else.")
        output = client.generate(inp)
        
        assert output.text is not None
        assert len(output.text) > 0
        assert output.latency_ms > 0
        assert output.input_tokens > 0
        assert output.output_tokens > 0

    def test_with_system_prompt(self, skip_if_no_gemini_key):
        """Test generation with system instruction."""
        client = GeminiClient(model_name="gemini-2.5-flash")
        
        inp = GenerationInput(
            system_prompt="You are a helpful assistant that only responds in JSON.",
            user_prompt='Return {"status": "ok"}',
        )
        output = client.generate(inp)
        
        assert output.text is not None
        assert "ok" in output.text.lower() or "status" in output.text.lower()

    def test_with_temperature(self, skip_if_no_gemini_key):
        """Test generation with custom temperature."""
        client = GeminiClient(model_name="gemini-2.5-flash")
        
        inp = GenerationInput(
            user_prompt="Say 'test' and nothing else.",
            temperature=1.0,  # Default for Gemini
        )
        output = client.generate(inp)
        
        assert output.text is not None
        assert len(output.text) > 0

    def test_with_max_tokens(self, skip_if_no_gemini_key):
        """Test generation with max output tokens limit."""
        client = GeminiClient(model_name="gemini-2.5-flash")
        
        inp = GenerationInput(
            user_prompt="Write a long story.",
            max_output_tokens=20,
        )
        output = client.generate(inp)
        
        assert output.text is not None
        assert output.output_tokens <= 25  # Allow small buffer

    def test_token_counting(self, skip_if_no_gemini_key):
        """Test Gemini API token counting."""
        client = GeminiClient(model_name="gemini-2.5-flash")
        
        text = "Hello, world!"
        count = client.count_tokens(text)
        
        assert count > 0
        assert count < 10  # Should be a few tokens

    def test_with_pricing(self, skip_if_no_gemini_key):
        """Test cost calculation with pricing configured."""
        pricing = PricingInfo(
            input_per_1k_tokens=0.000075,  # gemini-2.0-flash pricing (example)
            output_per_1k_tokens=0.0003,
        )
        client = GeminiClient(model_name="gemini-2.5-flash", pricing=pricing)
        
        inp = GenerationInput(user_prompt="Say hello.")
        output = client.generate(inp)
        
        assert output.cost_usd is not None
        assert output.cost_usd > 0
        assert output.cost_usd < 0.01  # Should be tiny cost

    def test_with_long_context(self, skip_if_no_gemini_key):
        """Test handling long context."""
        client = GeminiClient(model_name="gemini-2.5-flash")
        
        long_context = "Context: " + ("word " * 1000)
        inp = GenerationInput(
            context=long_context,
            user_prompt="What is the context about?",
        )
        output = client.generate(inp)
        
        assert output.text is not None
        assert output.input_tokens > 1000  # Should count context tokens


class TestScaleDownClientIntegration:
    """Integration tests for ScaleDown wrapper client with real API."""

    def test_compression_success(self, skip_if_no_scaledown_key, skip_if_no_gemini_key, wikipedia_context):
        """Test successful context compression with ScaleDown using Wikipedia data."""
        logger.info("=" * 80)
        logger.info("Test: Compression Success with Wikipedia")
        logger.info("=" * 80)
        
        # Create base Gemini client
        base_client = GeminiClient(model_name="gemini-2.5-flash")
        
        # Wrap with ScaleDown
        scaledown_client = ScaleDownClient(
            base_client=base_client,
            compression_rate="auto"
        )
        
        # Use Wikipedia content
        inp = GenerationInput(
            context=wikipedia_context,
            user_prompt="Who is Taylor Swift and what is she known for?",
        )
        
        logger.info(f"Context length: {len(wikipedia_context)} characters")
        logger.info(f"User prompt: {inp.user_prompt}")
        
        output = scaledown_client.generate(inp)
        
        logger.info(f"Response: {output.text[:200]}...")
        logger.info(f"Total latency: {output.latency_ms:.0f}ms")
        logger.info(f"Compression latency: {output.provider_metadata['compression_latency_ms']:.0f}ms")
        logger.info(f"Generation latency: {output.provider_metadata['generation_latency_ms']:.0f}ms")
        logger.info(f"Original tokens: {output.provider_metadata['original_prompt_tokens']}")
        logger.info(f"Compressed tokens: {output.provider_metadata['compressed_prompt_tokens']}")
        logger.info(f"Compression ratio: {output.provider_metadata['compression_ratio']:.2%}")
        
        # Verify output structure
        assert output.text is not None
        assert len(output.text) > 0
        assert output.latency_ms > 0
        assert output.input_tokens > 0
        assert output.output_tokens > 0
        
        # Verify ScaleDown metadata is present
        assert "scaledown_rate" in output.provider_metadata
        assert "compression_latency_ms" in output.provider_metadata
        assert "compression_ratio" in output.provider_metadata
        assert "original_prompt_tokens" in output.provider_metadata
        assert "compressed_prompt_tokens" in output.provider_metadata
        
        # Verify compression actually happened
        assert output.provider_metadata["compressed_prompt_tokens"] < output.provider_metadata["original_prompt_tokens"]

    def test_compression_metadata_capture(self, skip_if_no_scaledown_key, skip_if_no_gemini_key, wikipedia_context):
        """Test that compression metadata is properly captured."""
        logger.info("=" * 80)
        logger.info("Test: Compression Metadata Capture with Wikipedia")
        logger.info("=" * 80)
        
        base_client = GeminiClient(model_name="gemini-2.5-flash")
        scaledown_client = ScaleDownClient(base_client=base_client, compression_rate=0.5)
        
        inp = GenerationInput(
            context=wikipedia_context,
            user_prompt="What are some of Taylor Swift's most famous songs?",
        )
        
        logger.info(f"Compression rate: 0.5")
        logger.info(f"Context length: {len(wikipedia_context)} characters")
        logger.info(f"User prompt: {inp.user_prompt}")
        
        output = scaledown_client.generate(inp)
        
        logger.info(f"Response: {output.text[:200]}...")
        logger.info(f"Total latency: {output.latency_ms:.0f}ms")
        logger.info(f"Compression ratio: {output.provider_metadata['compression_ratio']:.2%}")
        
        # Check all required metadata fields
        metadata = output.provider_metadata
        assert metadata["scaledown_rate"] == 0.5
        assert metadata["compression_latency_ms"] > 0
        assert metadata["generation_latency_ms"] > 0
        assert 0 < metadata["compression_ratio"] < 1
        assert metadata["base_provider"] == "gemini"
        assert metadata["base_model"] == "gemini-2.5-flash"

    def test_baseline_vs_compressed(self, skip_if_no_scaledown_key, skip_if_no_gemini_key, wikipedia_context):
        """Test baseline vs ScaleDown compressed comparison with Wikipedia."""
        logger.info("=" * 80)
        logger.info("Test: Baseline vs Compressed Comparison with Wikipedia")
        logger.info("=" * 80)
        
        # Create base client
        base_client = GeminiClient(model_name="gemini-2.5-flash")
        
        # Question to ask
        question = "In what year was Taylor Swift born?"
        
        # Test 1: Baseline (uncompressed)
        logger.info("\n--- BASELINE (Uncompressed) ---")
        baseline_inp = GenerationInput(
            context=wikipedia_context,
            user_prompt=question,
        )
        
        baseline_output = base_client.generate(baseline_inp)
        
        logger.info(f"Response: {baseline_output.text}")
        logger.info(f"Input tokens: {baseline_output.input_tokens}")
        logger.info(f"Output tokens: {baseline_output.output_tokens}")
        logger.info(f"Latency: {baseline_output.latency_ms:.0f}ms")
        if baseline_output.cost_usd:
            logger.info(f"Estimated cost: ${baseline_output.cost_usd:.6f}")
        
        # Test 2: ScaleDown compressed
        logger.info("\n--- SCALEDOWN (Compressed) ---")
        scaledown_client = ScaleDownClient(
            base_client=base_client,
            compression_rate="auto"
        )
        
        compressed_inp = GenerationInput(
            context=wikipedia_context,
            user_prompt=question,
        )
        
        compressed_output = scaledown_client.generate(compressed_inp)
        
        logger.info(f"Response: {compressed_output.text}")
        logger.info(f"Input tokens: {compressed_output.input_tokens}")
        logger.info(f"Output tokens: {compressed_output.output_tokens}")
        logger.info(f"Latency: {compressed_output.latency_ms:.0f}ms")
        logger.info(f"Compression ratio: {compressed_output.provider_metadata['compression_ratio']:.2%}")
        if compressed_output.cost_usd:
            logger.info(f"Estimated cost: ${compressed_output.cost_usd:.6f}")
        
        # Compare results
        logger.info("\n--- COMPARISON ---")
        token_saving = baseline_output.input_tokens - compressed_output.input_tokens
        latency_diff = baseline_output.latency_ms - compressed_output.latency_ms
        
        logger.info(f"Token saving: {token_saving} ({token_saving/baseline_output.input_tokens*100:.1f}%)")
        logger.info(f"Latency difference: {latency_diff:.0f}ms")
        
        # Verify compression reduced tokens
        assert compressed_output.provider_metadata["compressed_prompt_tokens"] < baseline_output.input_tokens

    def test_fallback_on_no_context(self, skip_if_no_scaledown_key, skip_if_no_gemini_key):
        """Test fallback to base client when no context provided."""
        logger.info("=" * 80)
        logger.info("Test: Fallback on No Context")
        logger.info("=" * 80)
        
        base_client = GeminiClient(model_name="gemini-2.5-flash")
        scaledown_client = ScaleDownClient(base_client=base_client)
        
        # No context - should fallback
        inp = GenerationInput(user_prompt="Say hello.")
        logger.info("Generating without context (should fallback to base client)")
        
        output = scaledown_client.generate(inp)
        
        logger.info(f"Response: {output.text}")
        logger.info("Fallback successful")
        
        assert output.text is not None
        # Should not have ScaleDown-specific metadata since it fell back
        assert "scaledown_rate" not in output.provider_metadata or \
               output.provider_metadata.get("scaledown_rate") is None

    def test_token_counting_uses_base_client(self, skip_if_no_scaledown_key, skip_if_no_gemini_key, wikipedia_context):
        """Test that token counting delegates to base client."""
        logger.info("=" * 80)
        logger.info("Test: Token Counting")
        logger.info("=" * 80)
        
        base_client = GeminiClient(model_name="gemini-2.5-flash")
        scaledown_client = ScaleDownClient(base_client=base_client)
        
        count = scaledown_client.count_tokens(wikipedia_context)
        base_count = base_client.count_tokens(wikipedia_context)
        
        logger.info(f"Wikipedia content token count: {count}")
        logger.info(f"Base client count: {base_count}")
        
        assert count > 0
        assert count == base_count

    def test_with_pricing(self, skip_if_no_scaledown_key, skip_if_no_gemini_key, wikipedia_context):
        """Test cost calculation with ScaleDown compression using Wikipedia."""
        logger.info("=" * 80)
        logger.info("Test: Cost Calculation with Pricing")
        logger.info("=" * 80)
        
        pricing = PricingInfo(
            input_per_1k_tokens=0.000075,
            output_per_1k_tokens=0.0003,
        )
        base_client = GeminiClient(model_name="gemini-2.5-flash", pricing=pricing)
        scaledown_client = ScaleDownClient(base_client=base_client)
        
        inp = GenerationInput(
            context=wikipedia_context,
            user_prompt="Summarize this content in 2 sentences.",
        )
        
        logger.info("Generating with pricing enabled")
        output = scaledown_client.generate(inp)
        
        logger.info(f"Response: {output.text}")
        logger.info(f"Input tokens: {output.input_tokens}")
        logger.info(f"Output tokens: {output.output_tokens}")
        logger.info(f"Estimated cost: ${output.cost_usd:.6f}")
        
        # Cost should be calculated based on compressed tokens
        assert output.cost_usd is not None
        assert output.cost_usd > 0
