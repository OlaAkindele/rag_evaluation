import pandas as pd
from typing import List

def normalize_score(score: int, max_score: int = 5) -> float:
    """
    Normalizes a score to a 0-1 scale.
    
    Parameters:
        score (int): The raw score.
        max_score (int): The maximum possible score.
    
    Returns:
        float: Normalized score.
    """
    return score / max_score

def calculate_weighted_accuracy(scores: List[float], weights: List[float]) -> float:
    """
    Calculates weighted accuracy given normalized scores and corresponding weights.
    
    Parameters:
        scores (List[float]): List of normalized scores.
        weights (List[float]): List of weights corresponding to each score.
    
    Returns:
        float: The weighted accuracy.
    """
    weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
    total_weight = sum(weights)
    return weighted_sum / total_weight

def generate_report(scores: List[int], weights: List[float], metric_names: List[str], max_score: int = 5) -> pd.DataFrame:
    """
    Generates a classification-style report for evaluation metrics.
    
    Parameters:
        scores (List[int]): List of raw scores.
        weights (List[float]): List of weights corresponding to each metric.
        metric_names (List[str]): List of metric names.
        max_score (int): Maximum score for normalization.
    
    Returns:
        pd.DataFrame: A DataFrame report with normalized scores and percentage scores.
    """
    normalized_scores = [normalize_score(score, max_score) for score in scores]
    overall_accuracy = calculate_weighted_accuracy(normalized_scores, weights)
    
    report_data = {
        "Metric": metric_names + ["Overall Accuracy"],
        "Score (Normalized)": normalized_scores + [overall_accuracy],
        "Score (%)": [score * 100 for score in normalized_scores] + [overall_accuracy * 100],
    }
    return pd.DataFrame(report_data)
