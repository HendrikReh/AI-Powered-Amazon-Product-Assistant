"""
Evaluation module for AI-Powered Amazon Product Assistant.
Provides Ragas-based evaluation framework for RAG system.
"""

# New ragas-based evaluation
from .ragas_evaluator import RagasEvaluator, RagasEvaluationResult
from .rag_adapter import RAGAdapter, RagasEvaluationInput
from .ragas_reporter import RagasReporter

# Legacy evaluation (preserved for compatibility)
try:
    from .legacy.evaluator import RAGEvaluator, EvaluationMetrics
    from .legacy.scorers import (
        RelevanceScorer,
        AccuracyScorer,
        CompletenessScorer,
        FactualityScorer,
        ResponseQualityScorer
    )
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False

# Dataset utilities
try:
    from .dataset import create_evaluation_dataset, EvaluationExample
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False

__all__ = [
    # New ragas-based evaluation
    "RagasEvaluator",
    "RagasEvaluationResult",
    "RAGAdapter", 
    "RagasEvaluationInput",
    "RagasReporter"
]

# Add legacy items if available
if LEGACY_AVAILABLE:
    __all__.extend([
        "RAGEvaluator",
        "EvaluationMetrics",
        "RelevanceScorer",
        "AccuracyScorer", 
        "CompletenessScorer",
        "FactualityScorer",
        "ResponseQualityScorer"
    ])

# Add dataset items if available
if DATASET_AVAILABLE:
    __all__.extend([
        "create_evaluation_dataset",
        "EvaluationExample"
    ])