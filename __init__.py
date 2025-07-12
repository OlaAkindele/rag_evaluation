"""
rag_evaluation package

Provides tools for evaluating Retrieval-Augmented Generation (RAG) projects
by scoring generated responses across various metrics.

Currently, the default evaluator is based on GPT. In future, additional evaluators
(such as Gemini or Llama) will be added.
"""

# Import and re-export all public functionality from the gpt_evaluator sub-package.
from .rag_evaluation import (
    get_api_key,
    evaluate,
    evaluate_all,
    evaluate_response,
    normalize_score,
    calculate_weighted_accuracy,
    generate_report,
    EVALUATION_PROMPT_TEMPLATE,
    QUERY_RELEVANCE_CRITERIA,
    QUERY_RELEVANCE_STEPS,
    FACTUAL_ACCURACY_CRITERIA,
    FACTUAL_ACCURACY_STEPS,
    COVERAGE_CRITERIA,
    COVERAGE_STEPS,
    COHERENCE_SCORE_CRITERIA,
    COHERENCE_SCORE_STEPS,
    FLUENCY_SCORE_CRITERIA,
    FLUENCY_SCORE_STEPS,
)

# define a version string or other metadata.
__version__ = "0.2.0"

__all__ = [
    "get_api_key",
    "evaluate",
    "evaluate_all",
    "evaluate_response",
    "normalize_score",
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
