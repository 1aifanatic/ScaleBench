"""ROUGE metrics for abstractive summarization evaluation.

Implements ROUGE-1, ROUGE-2, and ROUGE-L using the HuggingFace evaluate library.
Follows the same signature as compute_bertscore in generation.py.

Dependencies: rouge-score, nltk, absl-py (transitive via evaluate).
"""
from typing import List, Dict, Callable, Optional
import evaluate as hf_evaluate


def _compute_rouge_all(
    predictions: List[str],
    references: List[List[str]],
    normalize_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, float]:
    """Shared helper: compute all ROUGE variants at once.

    Args:
        predictions: List of predicted summary strings.
        references: List of reference string lists (multiple refs per example).
        normalize_fn: Optional normalization function applied before scoring.
                      For ROUGE, this should only lowercase (no article stripping)
                      to preserve n-gram overlap.

    Returns:
        Dictionary with rouge1, rouge2, rougeL scores (macro-averaged).
    """
    metric = hf_evaluate.load("rouge")

    if normalize_fn:
        predictions = [normalize_fn(pred) for pred in predictions]
        references = [[normalize_fn(ref) for ref in refs] for refs in references]

    # evaluate's rouge metric accepts a list of strings or list of list of strings
    # for references; pass the first reference if only one exists, else the full list.
    flat_references = [refs[0] if len(refs) == 1 else refs for refs in references]

    result = metric.compute(
        predictions=predictions,
        references=flat_references,
        use_stemmer=True,
    )
    return result  # keys: rouge1, rouge2, rougeL, rougeLsum


def compute_rouge1(
    predictions: List[str],
    references: List[List[str]],
    normalize_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, float]:
    """Compute ROUGE-1 (unigram overlap).

    Args:
        predictions: List of predicted summary strings.
        references: List of reference string lists.
        normalize_fn: Optional normalization function.

    Returns:
        Dictionary with 'rouge1' score.
    """
    result = _compute_rouge_all(predictions, references, normalize_fn)
    return {"rouge1": result["rouge1"]}


def compute_rouge2(
    predictions: List[str],
    references: List[List[str]],
    normalize_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, float]:
    """Compute ROUGE-2 (bigram overlap).

    Args:
        predictions: List of predicted summary strings.
        references: List of reference string lists.
        normalize_fn: Optional normalization function.

    Returns:
        Dictionary with 'rouge2' score.
    """
    result = _compute_rouge_all(predictions, references, normalize_fn)
    return {"rouge2": result["rouge2"]}


def compute_rougeL(
    predictions: List[str],
    references: List[List[str]],
    normalize_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, float]:
    """Compute ROUGE-L (longest common subsequence).

    Args:
        predictions: List of predicted summary strings.
        references: List of reference string lists.
        normalize_fn: Optional normalization function.

    Returns:
        Dictionary with 'rougeL' score.
    """
    result = _compute_rouge_all(predictions, references, normalize_fn)
    return {"rougeL": result["rougeL"]}
