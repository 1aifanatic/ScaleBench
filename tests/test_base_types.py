"""Tests for base model types and interfaces."""
import pytest

from models.base import PricingInfo, GenerationInput, ModelOutput, ModelClient


class TestPricingInfo:
    """Tests for PricingInfo dataclass."""

    def test_pricing_info_creation(self):
        """Test creating a PricingInfo instance."""
        pricing = PricingInfo(input_per_1k_tokens=0.01, output_per_1k_tokens=0.03)
        assert pricing.input_per_1k_tokens == 0.01
        assert pricing.output_per_1k_tokens == 0.03

    def test_pricing_info_optional(self):
        """Test that pricing can be None."""
        pricing = PricingInfo()
        assert pricing.input_per_1k_tokens is None
        assert pricing.output_per_1k_tokens is None

    def test_calculate_cost_with_pricing(self):
        """Test cost calculation when pricing is set."""
        pricing = PricingInfo(input_per_1k_tokens=0.01, output_per_1k_tokens=0.03)
        cost = pricing.calculate_cost(input_tokens=1000, output_tokens=500)
        assert cost == pytest.approx(0.01 + 0.015)  # 1k*0.01 + 0.5k*0.03

    def test_calculate_cost_without_pricing(self):
        """Test cost calculation returns None when pricing not set."""
        pricing = PricingInfo()
        cost = pricing.calculate_cost(input_tokens=1000, output_tokens=500)
        assert cost is None

    def test_calculate_cost_partial_pricing(self):
        """Test cost calculation with only input pricing set."""
        pricing = PricingInfo(input_per_1k_tokens=0.01)
        cost = pricing.calculate_cost(input_tokens=1000, output_tokens=500)
        assert cost is None  # Both must be set


class TestGenerationInput:
    """Tests for GenerationInput dataclass."""

    def test_generation_input_minimal(self):
        """Test creating GenerationInput with only required fields."""
        inp = GenerationInput(user_prompt="Hello, world!")
        assert inp.user_prompt == "Hello, world!"
        assert inp.context is None
        assert inp.system_prompt is None
        assert inp.max_output_tokens is None
        assert inp.temperature is None
        assert inp.top_p is None
        assert inp.stop is None

    def test_generation_input_with_context(self):
        """Test GenerationInput with context for compression."""
        inp = GenerationInput(
            user_prompt="What is the answer?",
            context="Long context here...",
        )
        assert inp.user_prompt == "What is the answer?"
        assert inp.context == "Long context here..."

    def test_generation_input_with_all_params(self):
        """Test GenerationInput with all parameters."""
        inp = GenerationInput(
            user_prompt="Generate code",
            context="Documentation...",
            system_prompt="You are a coding assistant",
            max_output_tokens=500,
            temperature=0.7,
            top_p=0.9,
            stop=["###", "END"],
        )
        assert inp.user_prompt == "Generate code"
        assert inp.context == "Documentation..."
        assert inp.system_prompt == "You are a coding assistant"
        assert inp.max_output_tokens == 500
        assert inp.temperature == 0.7
        assert inp.top_p == 0.9
        assert inp.stop == ["###", "END"]


class TestModelOutput:
    """Tests for ModelOutput class."""

    def test_model_output_minimal(self):
        """Test creating ModelOutput with minimal fields."""
        output = ModelOutput(text="Generated text")
        assert output.text == "Generated text"
        assert output.latency_ms == 0.0
        assert output.input_tokens == 0
        assert output.output_tokens == 0
        assert output.cost_usd is None
        assert output.provider_metadata == {}
        assert output.metadata == {}

    def test_model_output_with_tokens_and_cost(self):
        """Test ModelOutput with token counts and cost."""
        output = ModelOutput(
            text="Response",
            latency_ms=123.45,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.0025,
        )
        assert output.text == "Response"
        assert output.latency_ms == 123.45
        assert output.input_tokens == 100
        assert output.output_tokens == 50
        assert output.cost_usd == 0.0025

    def test_model_output_with_provider_metadata(self):
        """Test ModelOutput with provider-specific metadata."""
        output = ModelOutput(
            text="Response",
            provider_metadata={
                "scaledown_rate": 0.8,
                "compression_ratio": 0.3,
                "compression_latency_ms": 50.0,
            },
        )
        assert output.provider_metadata["scaledown_rate"] == 0.8
        assert output.provider_metadata["compression_ratio"] == 0.3
        assert output.provider_metadata["compression_latency_ms"] == 50.0

    def test_model_output_backward_compatibility(self):
        """Test that metadata field is preserved for backward compatibility."""
        output = ModelOutput(
            text="Response",
            metadata={"custom_field": "value"},
        )
        assert output.metadata["custom_field"] == "value"
        assert output.provider_metadata == {}


class TestModelClientInterface:
    """Tests for ModelClient abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that ModelClient cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ModelClient(model_name="test-model")

    def test_concrete_implementation(self):
        """Test a concrete implementation of ModelClient."""
        
        class DummyClient(ModelClient):
            def __init__(self, model_name: str, **kwargs):
                super().__init__(model_name, **kwargs)
                self.provider = "dummy"
            
            def generate(self, inp: GenerationInput) -> ModelOutput:
                return ModelOutput(
                    text=f"Echo: {inp.user_prompt}",
                    input_tokens=10,
                    output_tokens=5,
                )
            
            def count_tokens(self, text: str) -> int:
                return len(text.split())
        
        client = DummyClient(model_name="dummy-model")
        assert client.model_name == "dummy-model"
        assert client.name == "dummy-model"
        assert client.provider == "dummy"
        assert client.pricing is None

    def test_client_with_pricing(self):
        """Test ModelClient with pricing information."""
        
        class DummyClient(ModelClient):
            def __init__(self, model_name: str, **kwargs):
                super().__init__(model_name, **kwargs)
                self.provider = "dummy"
            
            def generate(self, inp: GenerationInput) -> ModelOutput:
                cost = self._calculate_cost(10, 5)
                return ModelOutput(
                    text=f"Echo: {inp.user_prompt}",
                    input_tokens=10,
                    output_tokens=5,
                    cost_usd=cost,
                )
            
            def count_tokens(self, text: str) -> int:
                return len(text.split())
        
        pricing = PricingInfo(input_per_1k_tokens=0.01, output_per_1k_tokens=0.03)
        client = DummyClient(model_name="dummy-model", pricing=pricing)
        
        inp = GenerationInput(user_prompt="Test prompt")
        output = client.generate(inp)
        
        assert output.text == "Echo: Test prompt"
        assert output.cost_usd is not None
        assert output.cost_usd == pytest.approx(0.00025)  # (10/1000)*0.01 + (5/1000)*0.03

    def test_client_calculate_cost_helper(self):
        """Test the _calculate_cost helper method."""
        
        class DummyClient(ModelClient):
            def __init__(self, model_name: str, **kwargs):
                super().__init__(model_name, **kwargs)
            
            def generate(self, inp: GenerationInput) -> ModelOutput:
                return ModelOutput(text="test")
            
            def count_tokens(self, text: str) -> int:
                return 0
        
        # Without pricing
        client = DummyClient(model_name="test")
        assert client._calculate_cost(100, 50) is None
        
        # With pricing
        pricing = PricingInfo(input_per_1k_tokens=0.002, output_per_1k_tokens=0.006)
        client = DummyClient(model_name="test", pricing=pricing)
        cost = client._calculate_cost(1000, 500)
        assert cost == pytest.approx(0.002 + 0.003)  # 1k*0.002 + 0.5k*0.006
