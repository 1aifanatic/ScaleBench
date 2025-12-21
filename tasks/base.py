"""Task interface."""
from abc import ABC, abstractmethod
from typing import Dict, Any
import re
import string


class Task(ABC):
    """Abstract base class for tasks."""

    def __init__(self, name: str):
        """Initialize task.
        
        Args:
            name: Task name.
        """
        self.name = name

    @abstractmethod
    def build_prompt(self, example: Dict[str, Any], **kwargs) -> "GenerationInput":
        """Build GenerationInput from example.
        
        Args:
            example: Example dictionary from dataset.
            **kwargs: Additional generation parameters (temperature, etc.)
            
        Returns:
            GenerationInput object.
        """
        pass

    @abstractmethod
    def postprocess(self, output: str) -> str:
        """Extract and postprocess model output to get prediction.
        
        Args:
            output: Raw model output text.
            
        Returns:
            Extracted prediction string.
        """
        pass

    def normalize(self, text: str) -> str:
        """Normalize text for fair comparison.
        
        Default normalization: lowercase, remove articles, remove punctuation,
        collapse whitespace. Override in subclasses if needed.
        
        Args:
            text: Text to normalize.
            
        Returns:
            Normalized text.
        """
        # Lowercase
        text = text.lower()
        
        # Remove articles (a, an, the)
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Collapse whitespace
        text = ' '.join(text.split())
        
        return text.strip()

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name})"
