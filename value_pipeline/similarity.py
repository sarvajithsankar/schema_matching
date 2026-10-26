"""
Stage 4: Value similarity metrics for comparing column value profiles.
Supports weighted Euclidean distance, profile vector cosine similarity,
and Jensen-Shannon Divergence (JSD) distribution comparison.
All similarity metrics return values in [0.0, 1.0], are symmetric, deterministic,
and penalize data type mismatches.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.special import softmax

from value_pipeline.profiler import profile_to_vector

# Dimension weights corresponding to 22-dimensional profile vector:
# - Dims 0..4: dtype one-hot (weight 2.0 to strongly penalize dtype mismatch)
# - Dims 5..17: ratios, length stats, numeric stats (weight 1.0)
# - Dims 18..21: pattern flags (weight 1.5)
FEATURE_WEIGHTS = np.array(
    [2.0] * 5 +    # 0..4: dtype one-hot
    [1.0] * 13 +   # 5..17: ratios & statistics
    [1.5] * 4,     # 18..21: pattern flags
    dtype=np.float64
)


def weighted_euclidean_distance(profile_a: dict, profile_b: dict) -> float:
    """
    Computes weighted Euclidean distance between two profile vectors:
    dist = sqrt( sum( w_i * (v_a[i] - v_b[i])^2 ) )

    Symmetric: dist(a, b) == dist(b, a).
    Deterministic and handles zero vectors safely.
    """
    vec_a = np.array(profile_to_vector(profile_a), dtype=np.float64)
    vec_b = np.array(profile_to_vector(profile_b), dtype=np.float64)

    diff_sq = (vec_a - vec_b) ** 2
    weighted_diff = FEATURE_WEIGHTS * diff_sq
    return float(np.sqrt(np.sum(weighted_diff)))


def profile_cosine_similarity(profile_a: dict, profile_b: dict) -> float:
    """
    Computes cosine similarity between two 22-dimensional profile vectors,
    scaled to [0.0, 1.0].
    Handles zero vectors safely (returns 1.0 if both zero, 0.0 if one is zero).
    """
    vec_a = np.array(profile_to_vector(profile_a), dtype=np.float64)
    vec_b = np.array(profile_to_vector(profile_b), dtype=np.float64)

    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))

    if norm_a == 0.0 and norm_b == 0.0:
        return 1.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    dot = float(np.dot(vec_a, vec_b))
    cosine_val = dot / (norm_a * norm_b)
    # Map from [-1.0, 1.0] to [0.0, 1.0]
    scaled = (cosine_val + 1.0) / 2.0
    return float(np.clip(scaled, 0.0, 1.0))


def value_similarity(profile_a: dict, profile_b: dict) -> float:
    """
    Returns a similarity score in [0.0, 1.0] between two value_profile dicts.

    1. Vectorizes both profiles via profile_to_vector (22 dims).
    2. Computes weighted Euclidean distance (penalizing dtype mismatch).
    3. Transforms distance to similarity: sim = 1.0 / (1.0 + distance).
    4. Guarantees symmetry: value_similarity(a, b) == value_similarity(b, a).
    """
    dist = weighted_euclidean_distance(profile_a, profile_b)
    sim = 1.0 / (1.0 + dist)
    return float(np.clip(sim, 0.0, 1.0))


def value_similarity_jsd(profile_a: dict, profile_b: dict) -> float:
    """
    Jensen-Shannon divergence similarity for numeric columns.
    If both profiles have dtype == 'numeric' with numeric_stats, uses
    [q25, q50, q75, mean] to approximate distributions and returns 1.0 - JSD.
    Otherwise falls back to value_similarity().
    """
    dtype_a = profile_a.get("dtype")
    dtype_b = profile_b.get("dtype")

    if dtype_a in ("numeric", "int", "float") and dtype_b in ("numeric", "int", "float"):
        stats_a = profile_a.get("numeric_stats")
        stats_b = profile_b.get("numeric_stats")

        if isinstance(stats_a, dict) and isinstance(stats_b, dict):
            keys = ["q25", "q50", "q75", "mean"]
            vals_a = np.array([float(stats_a.get(k, 0.0) or 0.0) for k in keys], dtype=np.float64)
            vals_b = np.array([float(stats_b.get(k, 0.0) or 0.0) for k in keys], dtype=np.float64)

            # Convert to valid probability distributions using softmax
            p = softmax(vals_a)
            q = softmax(vals_b)

            js_dist = jensenshannon(p, q, base=2.0)
            if np.isnan(js_dist):
                return value_similarity(profile_a, profile_b)

            sim = max(0.0, min(1.0, 1.0 - float(js_dist)))
            return float(sim)

    # Fallback for non-numeric or missing statistics
    return value_similarity(profile_a, profile_b)
