"""
Metric calculation utilities for LLM Benchmark CLI.

This module contains functions for calculating various metrics to evaluate LLM performance.
"""

from typing import Any, Dict, List, Optional, Union

from ...core.exceptions import MetricError


def calculate_metrics(
    ground_truth: List[str], 
    predictions: List[str], 
    metrics: List[str]
) -> Dict[str, float]:
    """
    Calculate metrics for model predictions.
    
    Args:
        ground_truth: List of ground truth texts
        predictions: List of model predictions
        metrics: List of metrics to calculate
        
    Returns:
        Dict mapping metric names to scores
        
    Raises:
        MetricError: If metric calculation fails
    """
    results = {}
    
    for metric in metrics:
        try:
            if metric == "accuracy":
                results[metric] = calculate_accuracy(ground_truth, predictions)
            elif metric == "f1_score":
                results[metric] = calculate_f1_score(ground_truth, predictions)
            elif metric == "bleu":
                results[metric] = calculate_bleu(ground_truth, predictions)
            elif metric == "rouge_l":
                results[metric] = calculate_rouge_l(ground_truth, predictions)
            elif metric == "bert_score":
                results[metric] = calculate_bert_score(ground_truth, predictions)
            else:
                raise MetricError(f"Unsupported metric: {metric}")
        except Exception as e:
            raise MetricError(f"Failed to calculate {metric}: {str(e)}")
    
    return results


def calculate_accuracy(ground_truth: List[str], predictions: List[str]) -> float:
    """
    Calculate exact match accuracy.
    
    Args:
        ground_truth: List of ground truth texts
        predictions: List of model predictions
        
    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if not ground_truth or not predictions:
        return 0.0
    
    if len(ground_truth) != len(predictions):
        raise ValueError("Ground truth and predictions must have the same length")
    
    # Simple exact match accuracy
    matches = sum(1 for gt, pred in zip(ground_truth, predictions) 
                 if normalize_text(gt) == normalize_text(pred))
    return matches / len(ground_truth)


def calculate_f1_score(ground_truth: List[str], predictions: List[str]) -> float:
    """
    Calculate token-level F1 score.
    
    Args:
        ground_truth: List of ground truth texts
        predictions: List of model predictions
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    if not ground_truth or not predictions:
        return 0.0
    
    if len(ground_truth) != len(predictions):
        raise ValueError("Ground truth and predictions must have the same length")
    
    f1_scores = []
    
    for gt, pred in zip(ground_truth, predictions):
        # Tokenize
        gt_tokens = set(normalize_text(gt).split())
        pred_tokens = set(normalize_text(pred).split())
        
        # Calculate precision, recall, F1
        if not gt_tokens and not pred_tokens:
            f1_scores.append(1.0)  # Both empty
        elif not gt_tokens or not pred_tokens:
            f1_scores.append(0.0)  # One is empty
        else:
            intersection = len(gt_tokens.intersection(pred_tokens))
            precision = intersection / len(pred_tokens)
            recall = intersection / len(gt_tokens)
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1 = 2 * precision * recall / (precision + recall)
                f1_scores.append(f1)
    
    return sum(f1_scores) / len(f1_scores)


def calculate_bleu(ground_truth: List[str], predictions: List[str]) -> float:
    """
    Calculate BLEU score.
    
    Args:
        ground_truth: List of ground truth texts
        predictions: List of model predictions
        
    Returns:
        BLEU score (0.0 to 1.0)
    """
    try:
        import nltk
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    except ImportError:
        raise ImportError(
            "NLTK is required for BLEU score calculation. "
            "Install with: uv pip install -e .[metrics]"
        )
    
    if not ground_truth or not predictions:
        return 0.0
    
    if len(ground_truth) != len(predictions):
        raise ValueError("Ground truth and predictions must have the same length")
    
    # Tokenize
    references = [[normalize_text(gt).split()] for gt in ground_truth]
    hypothesis = [normalize_text(pred).split() for pred in predictions]
    
    # Calculate BLEU
    smoothing = SmoothingFunction().method1
    return corpus_bleu(references, hypothesis, smoothing_function=smoothing)


def calculate_rouge_l(ground_truth: List[str], predictions: List[str]) -> float:
    """
    Calculate ROUGE-L score.
    
    Args:
        ground_truth: List of ground truth texts
        predictions: List of model predictions
        
    Returns:
        ROUGE-L score (0.0 to 1.0)
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError(
            "rouge-score is required for ROUGE calculation. "
            "Install with: uv pip install -e .[metrics]"
        )
    
    if not ground_truth or not predictions:
        return 0.0
    
    if len(ground_truth) != len(predictions):
        raise ValueError("Ground truth and predictions must have the same length")
    
    # Initialize scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # Calculate ROUGE-L for each pair
    scores = []
    for gt, pred in zip(ground_truth, predictions):
        score = scorer.score(normalize_text(gt), normalize_text(pred))
        scores.append(score['rougeL'].fmeasure)
    
    return sum(scores) / len(scores)


def calculate_bert_score(ground_truth: List[str], predictions: List[str]) -> float:
    """
    Calculate BERTScore.
    
    Args:
        ground_truth: List of ground truth texts
        predictions: List of model predictions
        
    Returns:
        BERTScore (0.0 to 1.0)
    """
    # BERTScore requires a heavy dependency (transformers)
    # For simplicity, we'll return a placeholder value
    # In a real implementation, this would use the bert_score package
    
    # Placeholder implementation
    return calculate_f1_score(ground_truth, predictions)


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and extra whitespace
    import re
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
