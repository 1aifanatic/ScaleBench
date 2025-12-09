"""ModelClient interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class PricingInfo:
    """Pricing information for a model."""
    
    input_per_1k_tokens: Optional[float] = None
    output_per_1k_tokens: Optional[float] = None
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> Optional[float]:
        """Calculate cost in USD based on token usage.
        
        Args:
            input_tokens: Number of input tokens used.
            output_tokens: Number of output tokens generated.
            
        Returns:
            Cost in USD, or None if pricing not configured.
        """
        if self.input_per_1k_tokens is None or self.output_per_1k_tokens is None:
            return None
        
        input_cost = (input_tokens / 1000.0) * self.input_per_1k_tokens
        output_cost = (output_tokens / 1000.0) * self.output_per_1k_tokens
        return input_cost + output_cost


@dataclass
class GenerationInput:
    """Input for model generation with configurable parameters."""
    
    user_prompt: str
    context: Optional[str] = None
    system_prompt: Optional[str] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None


class ModelOutput:
    """Container for model generation output."""

    def __init__(
        self,
        text: str,
        latency_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: Optional[float] = None,
        provider_metadata: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.text = text
        self.latency_ms = latency_ms
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd
        self.provider_metadata = provider_metadata or {}
        # Keep metadata for backward compatibility
        self.metadata = metadata or {}


class ModelClient(ABC):
    """Abstract base class for model clients."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        pricing: Optional[PricingInfo] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.pricing = pricing
        self.provider = "unknown"  # Override in subclasses

    @abstractmethod
    def generate(self, inp: GenerationInput) -> ModelOutput:
        """Generate text from input.
        
        Args:
            inp: GenerationInput with prompt and configuration.
            
        Returns:
            ModelOutput with generated text and metadata.
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Input text to count tokens for.
            
        Returns:
            Token count.
        """
        pass

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> Optional[float]:
        """Calculate cost if pricing is configured.
        
        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            
        Returns:
            Cost in USD or None.
        """
        if self.pricing is None:
            return None
        return self.pricing.calculate_cost(input_tokens, output_tokens)

    @property
    def name(self) -> str:
        """Model name."""
        return self.model_name
