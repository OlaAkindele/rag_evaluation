"""
RAG Evaluation package

Provides tools for evaluating Retrieval-Augmented Generation (RAG) projects
by scoring generated responses across various metrics.
"""

from .config import get_openai_key
from .evaluator import (evaluate_all, evaluate_response,
                            evaluate_openai, evaluate_gemini,)
from .metrics import (
    EVALUATION_PROMPT_TEMPLATE,
    QUERY_RELEVANCE_CRITERIA, QUERY_RELEVANCE_STEPS,
    FACTUAL_ACCURACY_CRITERIA, FACTUAL_ACCURACY_STEPS,
    COVERAGE_CRITERIA, COVERAGE_STEPS,
    COHERENCE_SCORE_CRITERIA, COHERENCE_SCORE_STEPS,
    FLUENCY_SCORE_CRITERIA, FLUENCY_SCORE_STEPS,
)
from .utils import normalize_score, calculate_weighted_accuracy, generate_report

__all__ = [
    "get_openai_key",
    "evaluate",
    "evaluate_all",
    "evaluate_response",
    "normalize_score",
    "evaluate_openai",
    "evaluate_gemini",
    "calculate_weighted_accuracy",
    "generate_report",
    "EVALUATION_PROMPT_TEMPLATE",
    "QUERY_RELEVANCE_CRITERIA",
    "QUERY_RELEVANCE_STEPS",
    "FACTUAL_ACCURACY_CRITERIA",
    "FACTUAL_ACCURACY_STEPS",
    "COVERAGE_CRITERIA",
    "COVERAGE_STEPS",
    "COHERENCE_SCORE_CRITERIA",
    "COHERENCE_SCORE_STEPS",
    "FLUENCY_SCORE_CRITERIA",
    "FLUENCY_SCORE_STEPS",
]