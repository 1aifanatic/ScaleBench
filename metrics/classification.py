"""Classification metrics (F1, precision, recall, accuracy)."""
from typing import List, Dict, Callable, Optional
import evaluate as hf_evaluate


def compute_exact_match(
    predictions: List[str],
    references: List[List[str]],
    normalize_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, float]:
    """Compute exact match score using HuggingFace evaluate.
    
    Args:
        predictions: List of predicted strings.
        references: List of reference string lists (multiple refs per example).
        normalize_fn: Optional normalization function to apply before comparison.
        
    Returns:
        Dictionary with 'exact_match' score.
    """
    # Load metric lazily to avoid startup overhead
    metric = hf_evaluate.load("exact_match")
    
    # Flatten references to single strings (take first reference)
    # HF exact_match expects single strings, not lists
    flat_refs = [refs[0] if refs else "" for refs in references]
    
    # Apply normalization if provided
    if normalize_fn:
        predictions = [normalize_fn(pred) for pred in predictions]
        flat_refs = [normalize_fn(ref) for ref in flat_refs]
    
    # Compute metric
    result = metric.compute(predictions=predictions, references=flat_refs)
    
    return result


def compute_f1(
    predictions: List[str],
    references: List[List[str]],
    normalize_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, float]:
    """Compute token-level F1 score.
    
    For QA tasks, this computes token overlap F1 between prediction and references.
    Takes max F1 across multiple references.
    
    Args:
        predictions: List of predicted strings.
        references: List of reference string lists (multiple refs per example).
        normalize_fn: Optional normalization function to apply before comparison.
        
    Returns:
        Dictionary with 'f1' score.
    """
    
    total_f1 = 0.0
    
    for pred, refs in zip(predictions, references):
        if normalize_fn:
            pred = normalize_fn(pred)
            refs = [normalize_fn(ref) for ref in refs]
        
        # Compute F1 against each reference and take max
        max_f1 = 0.0
        for ref in refs:
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            if not pred_tokens or not ref_tokens:
                f1 = 0.0
            else:
                # Compute token overlap
                common = set(pred_tokens) & set(ref_tokens)
                num_common = len(common)
                
                if num_common == 0:
                    f1 = 0.0
                else:
                    precision = num_common / len(pred_tokens)
                    recall = num_common / len(ref_tokens)
                    f1 = 2 * (precision * recall) / (precision + recall)
            
            max_f1 = max(max_f1, f1)
        
        total_f1 += max_f1
    
    avg_f1 = total_f1 / len(predictions) if predictions else 0.0
    
    return {"f1": avg_f1}
