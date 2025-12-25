"""Set-based metrics for list/item evaluation (precision, recall, F1, exact match)."""
from typing import List, Dict, Set, Callable, Optional
import re


def _parse_items(text: str) -> Set[str]:
    """Parse model output into a set of normalized items.
    
    Handles multiple formats:
    - Newline-separated: "item1\\nitem2\\nitem3"
    - Comma-separated: "item1, item2, item3"
    - Mixed: "item1, item2\\nitem3, item4"
    - Bracketed items: "[123], [456]" or "[123]\\n[456]"
    
    Args:
        text: String containing items in various formats.
        
    Returns:
        Set of normalized item strings.
    """
    items = set()
    
    # First split by newlines, then by commas
    for line in text.split("\n"):
        # Split line by comma
        for part in line.split(","):
            item = part.strip()
            # Remove surrounding brackets if present
            if item.startswith("[") and item.endswith("]"):
                item = item[1:-1].strip()
            # Remove surrounding quotes if present
            if (item.startswith('"') and item.endswith('"')) or \
               (item.startswith("'") and item.endswith("'")):
                item = item[1:-1].strip()
            # Filter empty strings
            if item:
                items.add(item)
    return items


def _parse_reference(ref: List[str]) -> Set[str]:
    """Parse reference list into a set of normalized items.
    
    References from ETHIC are already lists of strings, but may have brackets.
    
    Args:
        ref: List of reference strings.
        
    Returns:
        Set of normalized item strings.
    """
    items = set()
    for item in ref:
        item = item.strip()
        # Remove surrounding brackets if present
        if item.startswith("[") and item.endswith("]"):
            item = item[1:-1].strip()
        if item:
            items.add(item)
    return items


def compute_set_exact_match(
    predictions: List[str],
    references: List[List[str]],
    normalize_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, float]:
    """Compute set-based exact match score.
    
    Score is 1.0 if predicted set exactly equals reference set, else 0.0.
    
    Args:
        predictions: List of predicted strings (multi-line, one item per line).
        references: List of reference string lists.
        normalize_fn: Optional normalization function (not used for set metrics).
        
    Returns:
        Dictionary with 'set_exact_match' score.
    """
    total_score = 0.0
    
    for pred, refs in zip(predictions, references):
        pred_set = _parse_items(pred)
        ref_set = _parse_reference(refs)
        
        if pred_set == ref_set:
            total_score += 1.0
    
    avg_score = total_score / len(predictions) if predictions else 0.0
    return {"set_exact_match": avg_score}


def compute_set_precision(
    predictions: List[str],
    references: List[List[str]],
    normalize_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, float]:
    """Compute set-based precision score.
    
    Precision = |predicted ∩ reference| / |predicted|
    Penalizes hallucinated items.
    
    Args:
        predictions: List of predicted strings (multi-line, one item per line).
        references: List of reference string lists.
        normalize_fn: Optional normalization function (not used for set metrics).
        
    Returns:
        Dictionary with 'set_precision' score.
    """
    total_precision = 0.0
    
    for pred, refs in zip(predictions, references):
        pred_set = _parse_items(pred)
        ref_set = _parse_reference(refs)
        
        if len(pred_set) == 0:
            precision = 0.0
        else:
            common = pred_set & ref_set
            precision = len(common) / len(pred_set)
        
        total_precision += precision
    
    avg_precision = total_precision / len(predictions) if predictions else 0.0
    return {"set_precision": avg_precision}


def compute_set_recall(
    predictions: List[str],
    references: List[List[str]],
    normalize_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, float]:
    """Compute set-based recall score.
    
    Recall = |predicted ∩ reference| / |reference|
    Penalizes missed items.
    
    Args:
        predictions: List of predicted strings (multi-line, one item per line).
        references: List of reference string lists.
        normalize_fn: Optional normalization function (not used for set metrics).
        
    Returns:
        Dictionary with 'set_recall' score.
    """
    total_recall = 0.0
    
    for pred, refs in zip(predictions, references):
        pred_set = _parse_items(pred)
        ref_set = _parse_reference(refs)
        
        if len(ref_set) == 0:
            recall = 0.0
        else:
            common = pred_set & ref_set
            recall = len(common) / len(ref_set)
        
        total_recall += recall
    
    avg_recall = total_recall / len(predictions) if predictions else 0.0
    return {"set_recall": avg_recall}


def compute_set_f1(
    predictions: List[str],
    references: List[List[str]],
    normalize_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, float]:
    """Compute set-based F1 score.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    Balances precision and recall.
    
    Args:
        predictions: List of predicted strings (multi-line, one item per line).
        references: List of reference string lists.
        normalize_fn: Optional normalization function (not used for set metrics).
        
    Returns:
        Dictionary with 'set_f1' score.
    """
    total_f1 = 0.0
    
    for pred, refs in zip(predictions, references):
        pred_set = _parse_items(pred)
        ref_set = _parse_reference(refs)
        
        # Compute precision
        if len(pred_set) == 0:
            precision = 0.0
        else:
            common = pred_set & ref_set
            precision = len(common) / len(pred_set)
        
        # Compute recall
        if len(ref_set) == 0:
            recall = 0.0
        else:
            common = pred_set & ref_set
            recall = len(common) / len(ref_set)
        
        # Compute F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        total_f1 += f1
    
    avg_f1 = total_f1 / len(predictions) if predictions else 0.0
    return {"set_f1": avg_f1}
