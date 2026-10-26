"""
Baseline matchers for comparative schema-matching evaluation.
Implements:
1. baseline_name_only: pure semantic name embedding / token similarity (alpha=1.0).
2. baseline_value_only: pure value distribution profiling similarity (alpha=0.0).
3. baseline_fixed_fusion: static fixed-weight fusion (alpha=0.5).
4. adaptive_matcher: dynamic adaptive fusion using Person A's confidence weighting.
"""

from typing import Any, List
import numpy as np

from matching.matcher import match_schemas
from person_a.name_similarity import compute_cosine_similarity
from person_a.normalize_names import normalize_column_name
from value_pipeline.similarity import value_similarity


def compute_name_similarity(profile_a: Any, profile_b: Any) -> float:
    """
    Computes cosine similarity between name_embedding vectors of two column profiles.
    Falls back to Person A's normalize_column_name token Jaccard similarity when
    embeddings are empty list [].
    """
    emb_a = (
        profile_a.get("name_embedding")
        if isinstance(profile_a, dict)
        else getattr(profile_a, "name_embedding", None)
    )
    emb_b = (
        profile_b.get("name_embedding")
        if isinstance(profile_b, dict)
        else getattr(profile_b, "name_embedding", None)
    )

    if emb_a is not None and emb_b is not None:
        arr_a = np.asarray(emb_a, dtype=np.float64)
        arr_b = np.asarray(emb_b, dtype=np.float64)
        if arr_a.size > 0 and arr_b.size > 0 and arr_a.shape == arr_b.shape:
            return float(compute_cosine_similarity(arr_a, arr_b))

    # Token overlap fallback using Person A's normalization pipeline
    tokens_a = (
        profile_a.get("normalized_tokens")
        if isinstance(profile_a, dict)
        else getattr(profile_a, "normalized_tokens", None)
    )
    tokens_b = (
        profile_b.get("normalized_tokens")
        if isinstance(profile_b, dict)
        else getattr(profile_b, "normalized_tokens", None)
    )

    if not tokens_a:
        raw_a = (
            profile_a.get("raw_name")
            if isinstance(profile_a, dict)
            else getattr(profile_a, "raw_name", None)
        )
        if not raw_a:
            raw_a = (
                profile_a.get("column_id", "").split(".")[-1]
                if isinstance(profile_a, dict)
                else getattr(profile_a, "column_id", "").split(".")[-1]
            )
        tokens_a = normalize_column_name(raw_a) if raw_a else []

    if not tokens_b:
        raw_b = (
            profile_b.get("raw_name")
            if isinstance(profile_b, dict)
            else getattr(profile_b, "raw_name", None)
        )
        if not raw_b:
            raw_b = (
                profile_b.get("column_id", "").split(".")[-1]
                if isinstance(profile_b, dict)
                else getattr(profile_b, "column_id", "").split(".")[-1]
            )
        tokens_b = normalize_column_name(raw_b) if raw_b else []

    set_a = set(tokens_a)
    set_b = set(tokens_b)

    if set_a and set_b:
        intersection = set_a & set_b
        union = set_a | set_b
        return float(len(intersection) / len(union)) if union else 0.0

    return 0.0


def baseline_name_only(
    profiles_a: List[Any],
    profiles_b: List[Any],
    threshold: float = 0.3
) -> List[Any]:
    """
    Name-only baseline: Uses only name similarity (alpha=1.0).
    """
    return match_schemas(
        profiles_a,
        profiles_b,
        name_sim_fn=compute_name_similarity,
        value_sim_fn=value_similarity,
        alpha=1.0,
        threshold=threshold,
        adaptive=False
    )


def baseline_value_only(
    profiles_a: List[Any],
    profiles_b: List[Any],
    threshold: float = 0.3
) -> List[Any]:
    """
    Value-only baseline: Uses only value_similarity (alpha=0.0).
    """
    return match_schemas(
        profiles_a,
        profiles_b,
        name_sim_fn=compute_name_similarity,
        value_sim_fn=value_similarity,
        alpha=0.0,
        threshold=threshold,
        adaptive=False
    )


def baseline_fixed_fusion(
    profiles_a: List[Any],
    profiles_b: List[Any],
    alpha: float = 0.5,
    threshold: float = 0.3
) -> List[Any]:
    """
    Fixed fusion baseline: Uses static alpha (default 0.5) combining name and value similarity.
    """
    return match_schemas(
        profiles_a,
        profiles_b,
        name_sim_fn=compute_name_similarity,
        value_sim_fn=value_similarity,
        alpha=alpha,
        threshold=threshold,
        adaptive=False
    )


def adaptive_matcher(
    profiles_a: List[Any],
    profiles_b: List[Any],
    threshold: float = 0.3
) -> List[Any]:
    """
    Adaptive fusion matcher using Person A's dynamic name/value fusion.
    """
    return match_schemas(
        profiles_a,
        profiles_b,
        name_sim_fn=compute_name_similarity,
        value_sim_fn=value_similarity,
        alpha=None,
        threshold=threshold,
        adaptive=True
    )
