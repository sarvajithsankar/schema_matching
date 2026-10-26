"""Unit tests for value_pipeline.profiler."""

import pytest
import numpy as np
from value_pipeline.profiler import (
    detect_dtype,
    infer_data_type,
    build_value_profile,
    profile_to_vector,
    PROFILE_VECTOR_FEATURE_MAP,
)


def test_profile_vector_feature_map_length():
    assert len(PROFILE_VECTOR_FEATURE_MAP) == 22
    assert PROFILE_VECTOR_FEATURE_MAP[0] == "is_numeric"
    assert PROFILE_VECTOR_FEATURE_MAP[21] == "pattern_date_str"


def test_detect_dtype_numeric():
    assert detect_dtype([1, 2, 3, 4, 5]) == "numeric"
    assert detect_dtype(["12.5", "44.0", "100", "0.05"]) == "numeric"


def test_infer_data_type_granular():
    assert infer_data_type([1, 2, 3, 4, 5]) == "int"
    assert infer_data_type([1.5, 2.5, 3.5]) == "float"
    assert infer_data_type([True, False, True, True]) == "bool"
    assert infer_data_type(["2023-01-15", "2023-02-20"]) == "date"
    assert infer_data_type(["C001", "EMP-042", "TXN_9981"]) == "id_code"
    assert infer_data_type(["ACTIVE"] * 40 + ["INACTIVE"] * 10) == "categorical"
    assert infer_data_type(["Quick brown fox", "Lorem ipsum"]) == "string"


def test_detect_dtype_date():
    dates = ["2023-01-15", "2023-02-20", "2023-03-25", "2023-04-30"]
    assert detect_dtype(dates) == "date"
    dates_slash = ["01/15/2023", "02/20/2023", "03/25/2023", "04/30/2023"]
    assert detect_dtype(dates_slash) == "date"


def test_detect_dtype_id_code():
    ids = ["C001", "EMP-042", "TXN_9981", "USR_1002", "ACC-774"]
    assert detect_dtype(ids) == "id_code"


def test_detect_dtype_categorical():
    vals = ["ACTIVE"] * 40 + ["INACTIVE"] * 10
    assert detect_dtype(vals) == "categorical"


def test_detect_dtype_string():
    texts = ["Quick brown fox", "Lorem ipsum dolor", "Database integration", "Semantic matching"]
    assert detect_dtype(texts) == "string"


def test_build_value_profile_keys():
    vals = ["alice@test.com", "bob@example.com", "carol@corp.org"]
    profile = build_value_profile(vals, total_count=5)

    assert "dtype" in profile
    assert "cardinality_ratio" in profile
    assert "null_ratio" in profile
    assert "length_stats" in profile
    assert "numeric_stats" in profile
    assert "pattern_flags" in profile

    assert profile["null_ratio"] == 0.4
    assert profile["pattern_flags"]["looks_like_email"] is True
    assert profile["pattern_flags"]["looks_like_phone"] is False


def test_build_value_profile_numeric_stats():
    nums = [10.0, 20.0, 30.0, 40.0, 50.0]
    profile = build_value_profile(nums)
    assert profile["dtype"] == "numeric"
    assert profile["numeric_stats"] is not None
    assert profile["numeric_stats"]["mean"] == 30.0
    assert profile["numeric_stats"]["min"] == 10.0
    assert profile["numeric_stats"]["max"] == 50.0
    assert profile["numeric_stats"]["q50"] == 30.0


def test_all_null_and_empty_columns():
    prof_empty = build_value_profile([])
    vec_empty = profile_to_vector(prof_empty)

    assert len(vec_empty) == 22
    assert all(np.isfinite(x) for x in vec_empty)
    assert not any(np.isnan(vec_empty))
    assert not any(np.isinf(vec_empty))

    prof_null = build_value_profile([], total_count=100)
    vec_null = profile_to_vector(prof_null)
    assert len(vec_null) == 22
    assert prof_null["null_ratio"] == 1.0
    assert all(np.isfinite(x) for x in vec_null)


def test_single_value_columns():
    prof_single_num = build_value_profile([42.0])
    vec_single_num = profile_to_vector(prof_single_num)

    assert len(vec_single_num) == 22
    assert all(np.isfinite(x) for x in vec_single_num)
    assert prof_single_num["numeric_stats"]["std"] == 0.0

    prof_single_str = build_value_profile(["only_one"])
    vec_single_str = profile_to_vector(prof_single_str)
    assert len(vec_single_str) == 22
    assert all(np.isfinite(x) for x in vec_single_str)


def test_profile_to_vector_dimensions_and_no_nan():
    vals_str = ["A10", "B20", "C30"]
    prof_str = build_value_profile(vals_str)
    vec_str = profile_to_vector(prof_str)

    vals_num = [1.5, 2.5, 3.5, 4.5]
    prof_num = build_value_profile(vals_num)
    vec_num = profile_to_vector(prof_num)

    assert isinstance(vec_str, list)
    assert isinstance(vec_num, list)
    assert len(vec_str) == 22
    assert len(vec_num) == 22
    assert all(isinstance(x, float) for x in vec_str)
    assert all(isinstance(x, float) for x in vec_num)
    assert all(np.isfinite(x) for x in vec_str)
    assert all(np.isfinite(x) for x in vec_num)
