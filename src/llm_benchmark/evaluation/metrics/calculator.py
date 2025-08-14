

from typing import Any, Dict, List, Optional, Union

from ...core.exceptions import MetricError

def calculate_metrics(
    ground_truth: List[str], 
    predictions: List[str], 
    metrics: List[str]
) -> Dict[str, float]:
    
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
    
    if not ground_truth or not predictions:
        return 0.0
    
    if len(ground_truth) != len(predictions):
        raise ValueError("Ground truth and predictions must have the same length")
    
    matches = sum(1 for gt, pred in zip(ground_truth, predictions) 
                 if normalize_text(gt) == normalize_text(pred))
    return matches / len(ground_truth)

def calculate_f1_score(ground_truth: List[str], predictions: List[str]) -> float:
    
    if not ground_truth or not predictions:
        return 0.0
    
    if len(ground_truth) != len(predictions):
        raise ValueError("Ground truth and predictions must have the same length")
    
    f1_scores = []
    
    for gt, pred in zip(ground_truth, predictions):
        gt_tokens = set(normalize_text(gt).split())
        pred_tokens = set(normalize_text(pred).split())
        
        if not gt_tokens and not pred_tokens:
            f1_scores.append(1.0)  
        elif not gt_tokens or not pred_tokens:
            f1_scores.append(0.0)  
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
    
    references = [[normalize_text(gt).split()] for gt in ground_truth]
    hypothesis = [normalize_text(pred).split() for pred in predictions]
    
    smoothing = SmoothingFunction().method1
    return corpus_bleu(references, hypothesis, smoothing_function=smoothing)

def calculate_rouge_l(ground_truth: List[str], predictions: List[str]) -> float:
    
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
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    scores = []
    for gt, pred in zip(ground_truth, predictions):
        score = scorer.score(normalize_text(gt), normalize_text(pred))
        scores.append(score['rougeL'].fmeasure)
    
    return sum(scores) / len(scores)

def calculate_bert_score(ground_truth: List[str], predictions: List[str]) -> float:
    
    
    return calculate_f1_score(ground_truth, predictions)

def normalize_text(text: str) -> str:
    
    text = text.lower()
    
    import re
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
