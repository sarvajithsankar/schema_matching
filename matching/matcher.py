"""
Stage 5: Schema Matcher using Hungarian algorithm (maximum weight bipartite matching).
Blends semantic name similarity and value similarity into a similarity matrix
using Person A's adaptive fusion (or baseline fixed fusion),
then finds the globally optimal one-to-one column correspondence.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import linear_sum_assignment

from person_a.fusion import fused_similarity
from person_a.name_similarity import compute_cosine_similarity
from person_a.normalize_names import normalize_column_name
from value_pipeline.similarity import value_similarity


class SchemaMatch(tuple):
    """
    Matched column pair represented as a 3-tuple (column_a, column_b, score)
    with dictionary compatibility for backwards compatibility with tests and metrics.
    """

    def __new__(cls, column_a: str, column_b: str, score: float, accepted: bool = True):
        return super().__new__(cls, (column_a, column_b, round(float(score), 4)))

    def __init__(self, column_a: str, column_b: str, score: float, accepted: bool = True):
        self.column_a = str(column_a)
        self.column_b = str(column_b)
        self.score = round(float(score), 4)
        self.accepted = bool(accepted)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            if item == "column_a":
                return self.column_a
            if item == "column_b":
                return self.column_b
            if item == "score":
                return self.score
            if item == "accepted":
                return self.accepted
            raise KeyError(item)
        return super().__getitem__(item)

    def get(self, key: str, default: Any = None) -> Any:
        if key in ("column_a", "column_b", "score", "accepted"):
            return getattr(self, key)
        return default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column_a": self.column_a,
            "column_b": self.column_b,
            "score": self.score,
            "accepted": self.accepted,
        }

    def __repr__(self) -> str:
        return f"SchemaMatch(column_a='{self.column_a}', column_b='{self.column_b}', score={self.score})"


def _get_column_identifier(profile: Any, fallback_idx: int) -> str:
    """Extract unique column ID or raw name from profile."""
    if hasattr(profile, "column_id") and profile.column_id:
        return str(profile.column_id)
    if isinstance(profile, dict):
        col_id = profile.get("column_id") or profile.get("raw_name")
        if col_id:
            return str(col_id)
    return f"col_{fallback_idx}"


def _get_column_name(profile: Any) -> str:
    """Extract raw column name for adaptive fusion name analysis."""
    if hasattr(profile, "raw_name") and profile.raw_name:
        return str(profile.raw_name)
    if isinstance(profile, dict):
        if profile.get("raw_name"):
            return str(profile["raw_name"])
        col_id = profile.get("column_id")
        if col_id:
            return str(col_id).split(".")[-1]
    if hasattr(profile, "column_id") and profile.column_id:
        return str(profile.column_id).split(".")[-1]
    return "column"


def _get_value_profile(profile: Any) -> dict:
    """Extract value_profile dict from ColumnProfile or dictionary."""
    if hasattr(profile, "value_profile"):
        return profile.value_profile
    if isinstance(profile, dict):
        return profile.get("value_profile", profile)
    return {}


def default_name_similarity(profile_a: Any, profile_b: Any) -> float:
    """
    Default name similarity:
    1. If dense name_embeddings are available, uses Person A's cosine similarity.
    2. Otherwise, uses Person A's normalize_column_name and Jaccard token similarity.
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
            return compute_cosine_similarity(arr_a, arr_b)

    # Token overlap fallback using Person A's normalizer
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
        tokens_a = normalize_column_name(_get_column_name(profile_a))
    if not tokens_b:
        tokens_b = normalize_column_name(_get_column_name(profile_b))

    set_a = set(tokens_a)
    set_b = set(tokens_b)

    if set_a and set_b:
        intersection = set_a & set_b
        union = set_a | set_b
        return float(len(intersection) / len(union)) if union else 0.0

    return 0.0


def build_similarity_matrix(
    profiles_a: List[Any],
    profiles_b: List[Any],
    name_sim_fn: Optional[Callable[[Any, Any], float]] = None,
    value_sim_fn: Optional[Callable[[Any, Any], float]] = None,
    alpha: Optional[float] = None,
    adaptive: bool = True
) -> np.ndarray:
    """
    Build an (|A| x |B|) similarity matrix.

    When alpha is explicitly provided and adaptive is False:
        entry [i][j] = alpha * name_sim + (1 - alpha) * value_sim
    When alpha is None or adaptive is True:
        entry [i][j] = fused_similarity(name_sim, value_sim, name_a, name_b)
        via Person A's adaptive fusion logic.
    """
    n_a = len(profiles_a)
    n_b = len(profiles_b)

    if n_a == 0 or n_b == 0:
        return np.zeros((n_a, n_b), dtype=np.float64)

    n_fn = name_sim_fn if name_sim_fn is not None else default_name_similarity
    v_fn = value_sim_fn if value_sim_fn is not None else value_similarity

    # If alpha is explicitly passed and adaptive is False, use fixed weighting
    use_adaptive = adaptive and (alpha is None)

    matrix = np.zeros((n_a, n_b), dtype=np.float64)

    for i, p_a in enumerate(profiles_a):
        vp_a = _get_value_profile(p_a)
        name_a = _get_column_name(p_a)

        for j, p_b in enumerate(profiles_b):
            vp_b = _get_value_profile(p_b)
            name_b = _get_column_name(p_b)

            n_sim = float(n_fn(p_a, p_b)) if n_fn is not None else 0.0
            v_sim = float(v_fn(vp_a, vp_b)) if v_fn is not None else 0.0

            if use_adaptive:
                score = fused_similarity(
                    name_sim=n_sim,
                    value_sim=v_sim,
                    source_name=name_a,
                    target_name=name_b
                )
            else:
                a_weight = alpha if alpha is not None else 0.5
                score = a_weight * n_sim + (1.0 - a_weight) * v_sim

            matrix[i, j] = float(np.clip(score, 0.0, 1.0))

    return matrix


def match_schemas(
    profiles_a: List[Any],
    profiles_b: List[Any],
    name_sim_fn: Optional[Callable[[Any, Any], float]] = None,
    value_sim_fn: Optional[Callable[[Any, Any], float]] = None,
    alpha: Optional[float] = None,
    threshold: float = 0.3,
    adaptive: bool = True
) -> List[SchemaMatch]:
    """
    Find globally optimal one-to-one column matches using the Hungarian algorithm
    (scipy.optimize.linear_sum_assignment) on the similarity matrix.

    Handles:
    - Rectangular matrices (|profiles_a| != |profiles_b|)
    - Empty schemas (|profiles_a| == 0 or |profiles_b| == 0)
    - Dropping matched pairs below confidence threshold

    Returns
    -------
    List[SchemaMatch]
        Sorted list of (column_a, column_b, score) match objects above threshold.
    """
    if not profiles_a or not profiles_b:
        return []

    sim_matrix = build_similarity_matrix(
        profiles_a,
        profiles_b,
        name_sim_fn=name_sim_fn,
        value_sim_fn=value_sim_fn,
        alpha=alpha,
        adaptive=adaptive
    )

    # Scipy minimizes cost; negate similarities for maximization problem
    cost_matrix = -sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches: List[SchemaMatch] = []
    for r, c in zip(row_ind, col_ind):
        score = float(sim_matrix[r, c])
        if score >= threshold:
            col_a_id = _get_column_identifier(profiles_a[r], r)
            col_b_id = _get_column_identifier(profiles_b[c], c)
            matches.append(SchemaMatch(col_a_id, col_b_id, score, accepted=True))

    # Sort descending by score
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches
