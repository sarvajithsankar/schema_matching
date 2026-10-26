"""
Schema matching package (Stage 5).
"""

from matching.matcher import build_similarity_matrix, match_schemas, SchemaMatch

__all__ = [
    "build_similarity_matrix",
    "match_schemas",
    "SchemaMatch",
]
