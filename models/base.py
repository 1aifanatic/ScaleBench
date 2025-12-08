"""ModelClient interface."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class ModelOutput:
    """Container for model generation output."""

    def __init__(
        self,
        text: str,
        latency_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.text = text
        self.latency_ms = latency_ms
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.metadata = metadata or {}


class ModelClient(ABC):
    """Abstract base class for model clients."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> ModelOutput:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt text.
            **kwargs: Additional model-specific parameters.
            
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

    @property
    def name(self) -> str:
        """Model name."""
        return self.model_name
