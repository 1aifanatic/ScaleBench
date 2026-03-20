"""Generation metrics (BLEU, ROUGE, BERTScore, etc.)."""
from typing import List, Dict, Callable, Optional
import evaluate as hf_evaluate


def compute_bertscore(
    predictions: List[str],
    references: List[List[str]],
    normalize_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, float]:
    """Compute BERTScore using HuggingFace evaluate.
    
    BERTScore uses BERT embeddings to compute semantic similarity between
    prediction and reference texts. Returns precision, recall, and F1 scores
    based on token-level contextual embeddings.
    
    Args:
        predictions: List of predicted strings.
        references: List of reference string lists (multiple refs per example).
        normalize_fn: Optional normalization function to apply before comparison.
        
    Returns:
        Dictionary with 'bertscore_precision', 'bertscore_recall', 'bertscore_f1'.
    """
    # Load metric lazily
    metric = hf_evaluate.load("bertscore")
    
    # Apply normalization if provided
    if normalize_fn:
        predictions = [normalize_fn(pred) for pred in predictions]
        references = [[normalize_fn(ref) for ref in refs] for refs in references]
    
    # Handle multiple references: compute BERTScore for each ref and take max
    all_precision = []
    all_recall = []
    all_f1 = []
    
    for pred, refs in zip(predictions, references):
        if not refs:
            all_precision.append(0.0)
            all_recall.append(0.0)
            all_f1.append(0.0)
            continue
        
        # Compute score against each reference
        ref_precision = []
        ref_recall = []
        ref_f1 = []
        
        for ref in refs:
            result = metric.compute(
                predictions=[pred],
                references=[ref],
                model_type="distilbert-base-uncased",
                lang="en"
            )
            # Extract scores for this prediction-reference pair
            ref_precision.append(result["precision"][0])
            ref_recall.append(result["recall"][0])
            ref_f1.append(result["f1"][0])
        
        # Take max scores across all references
        all_precision.append(max(ref_precision))
        all_recall.append(max(ref_recall))
        all_f1.append(max(ref_f1))
    
    # Return average BERTScore metrics across all predictions
    return {
        "bertscore_precision": sum(all_precision) / len(all_precision) if all_precision else 0.0,
        "bertscore_recall": sum(all_recall) / len(all_recall) if all_recall else 0.0,
        "bertscore_f1": sum(all_f1) / len(all_f1) if all_f1 else 0.0,
    }
