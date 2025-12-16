"""Simple registries for models/datasets/tasks/metrics."""
from typing import Dict, List, Callable
from core.config import ModelConfig, DatasetConfig, TaskConfig

# Import model clients
from models.openai_client import OpenAIClient
from models.gemini_client import GeminiClient
from models.scaledown_client import ScaleDownClient
from models.base import ModelClient, PricingInfo, DEFAULT_MODEL_PRICING

# Import datasets
from dataset.hotpotqa import HotpotQADataset
from dataset.base import Dataset

# Import tasks
from tasks.rag_task import RAGTask
from tasks.base import Task

# Import metrics
from metrics.classification import compute_exact_match, compute_f1


# ============================================================================
# REGISTRIES
# ============================================================================

MODEL_REGISTRY: Dict[str, type] = {
    "openai": OpenAIClient,
    "gemini": GeminiClient,
}

DATASET_REGISTRY: Dict[str, type] = {
    "hotpotqa": HotpotQADataset,
}

TASK_REGISTRY: Dict[str, type] = {
    "rag_qa": RAGTask,
}

METRIC_REGISTRY: Dict[str, Callable] = {
    "exact_match": compute_exact_match,
    "f1": compute_f1,
}


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def get_model_client(config: ModelConfig) -> ModelClient:
    """Instantiate model client from config.
    
    Args:
        config: ModelConfig with name and optional scaledown settings.
        
    Returns:
        ModelClient instance (baseline or wrapped with ScaleDownClient).
    """
    # Parse model name (format: "provider:model_name" or just "provider")
    parts = config.name.split(":")
    provider = parts[0].lower()
    model_name = parts[1] if len(parts) > 1 else None
    
    # Get base client class
    if provider not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model provider: {provider}. Available: {list(MODEL_REGISTRY.keys())}")
    
    client_class = MODEL_REGISTRY[provider]
    
    # Determine pricing (priority: YAML config > defaults > None)
    pricing = None
    if config.pricing:
        # User provided custom pricing in YAML
        pricing = PricingInfo(
            input_per_1m_tokens=config.pricing.get("input_per_1m_tokens"),
            output_per_1m_tokens=config.pricing.get("output_per_1m_tokens"),
        )
    elif model_name and model_name in DEFAULT_MODEL_PRICING:
        # Use default pricing for known model
        pricing = DEFAULT_MODEL_PRICING[model_name]
    
    # Instantiate base client with model name and pricing
    if model_name:
        base_client = client_class(model_name=model_name, pricing=pricing)
    else:
        base_client = client_class(pricing=pricing)
    
    # Wrap with ScaleDown if mode is 'scaledown'
    if config.mode == "scaledown":
        scaledown_config = config.scaledown or {}
        rate = scaledown_config.get("rate", "auto")
        compression_model = scaledown_config.get("compression_model")  # May be None
        return ScaleDownClient(
            base_client=base_client,
            compression_rate=rate,
            compression_model=compression_model,
            pricing=pricing,  # Pass pricing to ScaleDown wrapper
        )
    
    return base_client


def get_dataset(config: DatasetConfig) -> Dataset:
    """Instantiate dataset from config.
    
    Args:
        config: DatasetConfig with name and optional subset.
        
    Returns:
        Dataset instance.
    """
    # Parse dataset name (format: "dataset:subset" or just "dataset")
    parts = config.name.split(":")
    dataset_name = parts[0].lower()
    subset = parts[1] if len(parts) > 1 else None
    
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}")
    
    dataset_class = DATASET_REGISTRY[dataset_name]
    
    # Instantiate with subset if provided
    if subset:
        return dataset_class(name=config.name, subset=subset)
    else:
        return dataset_class(name=config.name)


def get_task(config: TaskConfig) -> Task:
    """Instantiate task from config.
    
    Args:
        config: TaskConfig with name.
        
    Returns:
        Task instance.
    """
    task_name = config.name.lower()
    
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_REGISTRY.keys())}")
    
    task_class = TASK_REGISTRY[task_name]
    return task_class(name=config.name)


def get_metrics(metric_names: List[str]) -> Dict[str, Callable]:
    """Get metric functions from names.
    
    Args:
        metric_names: List of metric names to load.
        
    Returns:
        Dictionary mapping metric names to callable functions.
    """
    metrics = {}
    
    for name in metric_names:
        name_lower = name.lower()
        if name_lower not in METRIC_REGISTRY:
            raise ValueError(f"Unknown metric: {name}. Available: {list(METRIC_REGISTRY.keys())}")
        metrics[name_lower] = METRIC_REGISTRY[name_lower]
    
    return metrics
