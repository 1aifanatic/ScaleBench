"""Data sampler for load testing - samples examples from datasets.

This module provides a thread-safe data sampler that pre-loads examples
from any dataset implementing the Dataset ABC and provides random sampling
for Locust load testing scenarios.
"""
import logging
import random

from dataset.base import Example
from core.registry import get_dataset
from core.config import DatasetConfig


logger = logging.getLogger(__name__)


class DataSampler:
    """Thread-safe data sampler for load testing.
    
    Pre-loads examples from a dataset at initialization and provides
    thread-safe random sampling for concurrent Locust users.
    
    Args:
        dataset_name: Name of dataset (must be registered in DATASET_REGISTRY)
        split: Dataset split (train/validation/test)
        num_samples: Number of examples to pre-load
        seed: Random seed for reproducible sampling
        
    Attributes:
        dataset: Dataset instance
        examples: List of pre-loaded examples
        name: Dataset name
        split: Dataset split
        num_loaded: Number of examples loaded
        context_length_range: Tuple of (min, max) context lengths in chars
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "validation",
        num_samples: int = 100,
        seed: int = 42,
    ):
        """Initialize data sampler and pre-load examples.
        
        Args:
            dataset_name: Name of dataset to load
            split: Dataset split to load from
            num_samples: Number of examples to pre-load
            seed: Random seed for reproducibility
        """
        self.name = dataset_name
        self.split = split
        self.seed = seed
        
        # Create dataset config and load dataset using registry
        dataset_config = DatasetConfig(name=dataset_name)
        self.dataset = get_dataset(dataset_config)
        
        logger.info(f"Loading {num_samples} examples from {dataset_name} ({split} split)...")
        
        # Pre-load examples into memory
        self.examples = []
        example_iterator = self.dataset.load_examples(
            split=split,
            limit=num_samples,
            seed=seed,
        )
        
        for example in example_iterator:
            self.examples.append(example)
        
        self.num_loaded = len(self.examples)
        
        # Calculate context length range
        if self.examples:
            context_lengths = [len(ex.get("context", "")) for ex in self.examples]
            self.context_length_range = (min(context_lengths), max(context_lengths))
        else:
            self.context_length_range = (0, 0)
        
        logger.info(
            f"Loaded {self.num_loaded} examples from {dataset_name}. "
            f"Context length range: {self.context_length_range[0]}-{self.context_length_range[1]} chars"
        )
        
        if self.num_loaded == 0:
            raise ValueError(f"No examples loaded from {dataset_name} ({split} split)")
    
    def get_random_example(self) -> Example:
        """Get a random example from pre-loaded pool.
        
        This method is thread-safe because random.choice() is thread-safe
        and we only read from the pre-loaded list.
        
        Returns:
            Random Example from the pre-loaded pool
            
        Raises:
            ValueError: If no examples are loaded
        """
        if not self.examples:
            raise ValueError("No examples available for sampling")
        
        return random.choice(self.examples)
    
    def get_stats(self) -> dict:
        """Get basic statistics about loaded data.
        
        Returns:
            Dictionary with dataset statistics
        """
        return {
            "dataset_name": self.name,
            "split": self.split,
            "num_examples": self.num_loaded,
            "context_length_min": self.context_length_range[0],
            "context_length_max": self.context_length_range[1],
            "seed": self.seed,
        }