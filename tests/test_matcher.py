"""Unit tests for matching.matcher."""

import pytest
import numpy as np

from matching.matcher import build_similarity_matrix, match_schemas, SchemaMatch
from evaluation.benchmark import load_synthetic_pairs
from evaluation.baselines import compute_name_similarity, adaptive_matcher
from value_pipeline.similarity import value_similarity
from value_pipeline.profiler import build_value_profile


def test_build_similarity_matrix():
    pairs = load_synthetic_pairs()
    assert len(pairs) > 0, "Synthetic pairs must be generated"

    pair_1 = pairs[0]
    profiles_a = pair_1["profiles_a"]
    profiles_b = pair_1["profiles_b"]

    matrix = build_similarity_matrix(
        profiles_a,
        profiles_b,
        name_sim_fn=compute_name_similarity,
        value_sim_fn=value_similarity,
        alpha=0.5
    )

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (len(profiles_a), len(profiles_b))
    assert (matrix >= 0.0).all() and (matrix <= 1.0).all()


def test_match_schemas_synthetic_pair_1():
    pairs = load_synthetic_pairs()
    pair_1 = pairs[0]
    assert pair_1["pair_id"] == "pair_1"

    matches = match_schemas(
        pair_1["profiles_a"],
        pair_1["profiles_b"],
        name_sim_fn=compute_name_similarity,
        value_sim_fn=value_similarity,
        alpha=0.5,
        threshold=0.3
    )

    assert len(matches) == 4
    for m in matches:
        assert m["accepted"] is True
        assert m["score"] > 0.8
        # Source ID and target ID should have matching suffix
        col_a_suffix = m["column_a"].split(".")[-1]
        col_b_suffix = m["column_b"].split(".")[-1]
        assert col_a_suffix == col_b_suffix

        # Test tuple unpacking as well
        ca, cb, s = m
        assert ca == m["column_a"]
        assert cb == m["column_b"]
        assert s == m["score"]


def test_match_schemas_rectangular_matrices():
    # Schema A has 2 columns, Schema B has 4 columns
    profiles_a = [
        {"column_id": "A.id", "raw_name": "id", "value_profile": build_value_profile(["C01", "C02"])},
        {"column_id": "A.email", "raw_name": "email", "value_profile": build_value_profile(["a@b.com", "c@d.com"])},
    ]
    profiles_b = [
        {"column_id": "B.id", "raw_name": "id", "value_profile": build_value_profile(["C01", "C02"])},
        {"column_id": "B.email", "raw_name": "email", "value_profile": build_value_profile(["a@b.com", "c@d.com"])},
        {"column_id": "B.phone", "raw_name": "phone", "value_profile": build_value_profile(["1234", "5678"])},
        {"column_id": "B.age", "raw_name": "age", "value_profile": build_value_profile([25, 30])},
    ]

    matches_a_to_b = match_schemas(profiles_a, profiles_b, threshold=0.3)
    # At most 2 matches can be selected (since |A| == 2)
    assert len(matches_a_to_b) <= 2
    matched_a = [m[0] for m in matches_a_to_b]
    matched_b = [m[1] for m in matches_a_to_b]
    # Check one-to-one constraint: all matched columns must be unique
    assert len(matched_a) == len(set(matched_a))
    assert len(matched_b) == len(set(matched_b))

    # Reverse: Schema A has 4 columns, Schema B has 2 columns
    matches_b_to_a = match_schemas(profiles_b, profiles_a, threshold=0.3)
    assert len(matches_b_to_a) <= 2
    assert len(set(m[0] for m in matches_b_to_a)) == len(matches_b_to_a)
    assert len(set(m[1] for m in matches_b_to_a)) == len(matches_b_to_a)


def test_match_schemas_threshold_dropping():
    # Completely incompatible columns
    profiles_a = [
        {"column_id": "A.amount", "raw_name": "amount", "value_profile": build_value_profile([100.0, 200.0])}
    ]
    profiles_b = [
        {"column_id": "B.birth_date", "raw_name": "birth_date", "value_profile": build_value_profile(["1990-01-01", "1995-05-05"])}
    ]

    # High threshold should drop the pair
    matches_high_thresh = match_schemas(profiles_a, profiles_b, threshold=0.9)
    assert len(matches_high_thresh) == 0


def test_match_schemas_empty_input():
    assert match_schemas([], []) == []
    assert match_schemas([{"column_id": "A.x"}], []) == []
    assert match_schemas([], [{"column_id": "B.y"}]) == []


def test_end_to_end_adaptive_matcher_synthetic():
    # Tiny synthetic pair
    schema_a = [
        {"column_id": "S.cust_id", "raw_name": "cust_id", "value_profile": build_value_profile(["C01", "C02", "C03"])},
        {"column_id": "S.email_addr", "raw_name": "email_addr", "value_profile": build_value_profile(["u1@test.com", "u2@test.com"])},
    ]
    schema_b = [
        {"column_id": "T.client_id", "raw_name": "client_id", "value_profile": build_value_profile(["C01", "C02", "C03"])},
        {"column_id": "T.contact_email", "raw_name": "contact_email", "value_profile": build_value_profile(["u1@test.com", "u2@test.com"])},
    ]

    matches = adaptive_matcher(schema_a, schema_b, threshold=0.3)
    assert len(matches) == 2
    match_dict = {m[0]: m[1] for m in matches}
    assert match_dict["S.cust_id"] == "T.client_id"
    assert match_dict["S.email_addr"] == "T.contact_email"
