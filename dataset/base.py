"""Dataset interface."""
from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, Optional, List


class Example(Dict[str, Any]):
    """Example from a dataset.
    
    Standard fields:
        id: Unique example identifier.
        context: Context text (passages, documents, etc.).
        question: Question or query text.
        answer: Ground truth answer(s) as list of strings.
    
    Additional fields can be added per dataset.
    """
    pass


class Dataset(ABC):
    """Abstract base class for datasets."""

    def __init__(self, name: str):
        """Initialize dataset.
        
        Args:
            name: Dataset name.
        """
        self.name = name

    @abstractmethod
    def load_examples(
        self,
        split: str = "test",
        limit: Optional[int] = None,
        seed: int = 42,
    ) -> Iterator[Example]:
        """Load and yield examples from dataset.
        
        Args:
            split: Dataset split (train/validation/test).
            limit: Maximum number of examples to load (None = all).
            seed: Random seed for deterministic sampling.
            
        Yields:
            Example dictionaries with at minimum {id, context, question, answer}.
            answer should always be a list of strings for consistency.
        """
        pass

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name})"
