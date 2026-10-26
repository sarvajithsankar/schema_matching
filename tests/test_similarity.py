"""Unit tests for value_pipeline.similarity."""

import pytest
import numpy as np
from value_pipeline.profiler import build_value_profile
from value_pipeline.similarity import (
    value_similarity,
    value_similarity_jsd,
    profile_cosine_similarity,
    weighted_euclidean_distance
)


def test_identical_profiles_high_similarity():
    vals = ["TXN-001", "TXN-002", "TXN-003", "TXN-004"]
    p1 = build_value_profile(vals)
    p2 = build_value_profile(vals)

    dist = weighted_euclidean_distance(p1, p2)
    assert dist == pytest.approx(0.0, abs=1e-5)

    sim = value_similarity(p1, p2)
    assert sim == pytest.approx(1.0, abs=1e-3)


def test_similarity_symmetry_and_range():
    p1 = build_value_profile([10, 20, 30, 40])
    p2 = build_value_profile(["alpha", "beta", "gamma"])

    sim_12 = value_similarity(p1, p2)
    sim_21 = value_similarity(p2, p1)

    assert sim_12 == pytest.approx(sim_21, abs=1e-7)
    assert 0.0 <= sim_12 <= 1.0
    assert 0.0 <= sim_21 <= 1.0


def test_zero_vector_handling():
    empty_p1 = build_value_profile([])
    empty_p2 = build_value_profile([])

    sim = value_similarity(empty_p1, empty_p2)
    assert 0.0 <= sim <= 1.0

    cos_sim = profile_cosine_similarity(empty_p1, empty_p2)
    assert 0.0 <= cos_sim <= 1.0


def test_dtype_mismatch_low_similarity():
    # Numeric column vs date strings
    nums = [10.0, 20.0, 30.0, 40.0]
    dates = ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]

    p_num = build_value_profile(nums)
    p_date = build_value_profile(dates)

    sim = value_similarity(p_num, p_date)
    # The requirement specifies dtype mismatch should strongly penalize, score < 0.4
    assert sim < 0.4


def test_value_similarity_jsd_numeric():
    nums1 = [10.0, 20.0, 30.0, 40.0, 50.0]
    nums2 = [10.5, 20.2, 29.8, 40.1, 49.9]

    p1 = build_value_profile(nums1)
    p2 = build_value_profile(nums2)

    sim_jsd = value_similarity_jsd(p1, p2)
    assert 0.0 <= sim_jsd <= 1.0
    assert sim_jsd > 0.8


def test_value_similarity_jsd_fallback_non_numeric():
    s1 = ["hello", "world"]
    s2 = ["foo", "bar"]
    p1 = build_value_profile(s1)
    p2 = build_value_profile(s2)

    sim_jsd = value_similarity_jsd(p1, p2)
    assert 0.0 <= sim_jsd <= 1.0
