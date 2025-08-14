from typing import Any, Dict, List, Optional, Union, Callable

import dspy
from dspy.evaluate import Evaluate

from ...core.exceptions import MetricError

def create_dspy_metric(metric_name: str) -> Callable:
    """
    Create a DSPy-compatible metric function based on the metric name.
    
    Args:
        metric_name: Name of the metric to create
        
    Returns:
        A metric function compatible with dspy.Evaluate
    """
    if metric_name == "accuracy":
        return accuracy_metric
    elif metric_name == "f1_score":
        return f1_score_metric
    elif metric_name == "contains":
        return contains_metric
    elif metric_name == "exact_match":
        return exact_match_metric
    elif metric_name == "semantic_similarity":
        return semantic_similarity_metric
    else:
        raise MetricError(f"Unsupported DSPy metric: {metric_name}")

def accuracy_metric(gold, pred, trace=None):
    """
    Simple accuracy metric for DSPy evaluation.
    
    Args:
        gold: Gold standard example
        pred: Model prediction
        trace: Optional trace of the model's execution
        
    Returns:
        Score between 0.0 and 1.0
    """
    if hasattr(gold, 'output') and hasattr(pred, 'output'):
        return 1.0 if gold.output.strip() == pred.output.strip() else 0.0
    return 0.0

def f1_score_metric(gold, pred, trace=None):
    """
    F1 score metric for DSPy evaluation.
    
    Args:
        gold: Gold standard example
        pred: Model prediction
        trace: Optional trace of the model's execution
        
    Returns:
        Score between 0.0 and 1.0
    """
    if not hasattr(gold, 'output') or not hasattr(pred, 'output'):
        return 0.0
    
    gt_tokens = set(_normalize_text(gold.output).split())
    pred_tokens = set(_normalize_text(pred.output).split())
    
    if not gt_tokens and not pred_tokens:
        return 1.0
    elif not gt_tokens or not pred_tokens:
        return 0.0
    
    intersection = len(gt_tokens.intersection(pred_tokens))
    precision = intersection / len(pred_tokens) if pred_tokens else 0.0
    recall = intersection / len(gt_tokens) if gt_tokens else 0.0
    
    if precision + recall == 0:
        return 0.0
    else:
        return 2 * precision * recall / (precision + recall)

def contains_metric(gold, pred, trace=None):
    """
    Check if prediction contains the gold answer.
    
    Args:
        gold: Gold standard example
        pred: Model prediction
        trace: Optional trace of the model's execution
        
    Returns:
        1.0 if prediction contains gold, 0.0 otherwise
    """
    if hasattr(gold, 'output') and hasattr(pred, 'output'):
        return 1.0 if _normalize_text(gold.output) in _normalize_text(pred.output) else 0.0
    return 0.0

def exact_match_metric(gold, pred, trace=None):
    """
    Exact match metric using dspy's answer_exact_match.
    
    Args:
        gold: Gold standard example
        pred: Model prediction
        trace: Optional trace of the model's execution
        
    Returns:
        Score between 0.0 and 1.0
    """
    try:
        from dspy.evaluate import answer_exact_match
        if hasattr(gold, 'output') and hasattr(pred, 'output'):
            return answer_exact_match(gold.output, pred.output)
        return 0.0
    except ImportError:
        # Fall back to simple accuracy if dspy.evaluate is not available
        return accuracy_metric(gold, pred, trace)

def semantic_similarity_metric(gold, pred, trace=None):
    """
    Semantic similarity metric using embeddings.
    
    Args:
        gold: Gold standard example
        pred: Model prediction
        trace: Optional trace of the model's execution
        
    Returns:
        Score between 0.0 and 1.0
    """
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        
        if not hasattr(gold, 'output') or not hasattr(pred, 'output'):
            return 0.0
        
        # Load the model (this should ideally be cached)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode the sentences
        gold_embedding = model.encode(_normalize_text(gold.output))
        pred_embedding = model.encode(_normalize_text(pred.output))
        
        # Calculate cosine similarity
        similarity = np.dot(gold_embedding, pred_embedding) / (
            np.linalg.norm(gold_embedding) * np.linalg.norm(pred_embedding)
        )
        
        return float(similarity)
    except ImportError:
        # Fall back to F1 score if sentence_transformers is not available
        return f1_score_metric(gold, pred, trace)

def evaluate_with_dspy(program, dataset, metric_name="accuracy", num_threads=1):
    """
    Evaluate a DSPy program using dspy.Evaluate.
    
    Args:
        program: The DSPy program to evaluate
        dataset: The dataset to evaluate on
        metric_name: The name of the metric to use
        num_threads: Number of threads to use for evaluation
        
    Returns:
        Evaluation results
    """
    # Get the appropriate metric function
    metric_fn = create_dspy_metric(metric_name)
    
    # Create evaluator
    evaluator = Evaluate(
        program,
        metric=metric_fn,
        num_threads=num_threads
    )
    
    # Run evaluation
    results = evaluator(dataset)
    
    return {
        "score": results.score,
        "results": results
    }

def _normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    text = text.lower()
    
    import re
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
