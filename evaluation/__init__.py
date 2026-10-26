"""
Evaluation package for schema matching benchmarks, baselines, and metrics.
"""

from evaluation.metrics import compute_metrics, compute_macro_metrics
from evaluation.baselines import (
    compute_name_similarity,
    baseline_name_only,
    baseline_value_only,
    baseline_fixed_fusion,
    adaptive_matcher,
)

__all__ = [
    "compute_metrics",
    "compute_macro_metrics",
    "compute_name_similarity",
    "baseline_name_only",
    "baseline_value_only",
    "baseline_fixed_fusion",
    "adaptive_matcher",
]
