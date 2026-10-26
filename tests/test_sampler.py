"""Unit tests for value_pipeline.sampler."""

import pytest
import tempfile
from pathlib import Path
import pandas as pd

from value_pipeline.sampler import sample_column, load_csv_column


def test_sample_column_less_than_n():
    vals = ["a", "b", "c"]
    sampled = sample_column(vals, n=10)
    assert len(sampled) == 3
    assert set(sampled) == {"a", "b", "c"}


def test_sample_column_more_than_n():
    vals = list(range(100))
    sampled = sample_column(vals, n=20, random_state=42)
    assert len(sampled) == 20
    assert len(set(sampled)) == 20
    assert all(x in vals for x in sampled)


def test_sample_column_filters_nulls():
    vals = ["valid", None, float("nan"), "null", "NaN", "another_valid", ""]
    sampled = sample_column(vals, n=10)
    assert "valid" in sampled
    assert "another_valid" in sampled
    assert None not in sampled
    assert "" not in sampled
    assert len(sampled) == 2


def test_sample_column_none_and_empty():
    assert sample_column(None, n=10) == []
    assert sample_column([], n=10) == []
    assert sample_column([None, "", "null", float("nan")], n=10) == []


def test_sample_column_huge():
    huge_vals = list(range(50000))
    sampled = sample_column(huge_vals, n=100, random_state=42)
    assert len(sampled) == 100
    assert len(set(sampled)) == 100


def test_sample_column_mixed_types():
    mixed = [1, "two", 3.14, True, None, "2023-01-01", False]
    sampled = sample_column(mixed, n=10, random_state=42)
    # Booleans (True, False) and numbers are valid non-nulls; None is filtered
    assert None not in sampled
    assert 1 in sampled
    assert "two" in sampled
    assert 3.14 in sampled
    assert True in sampled
    assert False in sampled


def test_sample_column_deterministic_seed():
    vals = list(range(1000))
    s1 = sample_column(vals, n=50, random_state=123)
    s2 = sample_column(vals, n=50, random_state=123)
    s3 = sample_column(vals, n=50, random_state=999)
    assert s1 == s2
    assert s1 != s3


def test_load_csv_column():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("col_a,col_b\nval1,10\nval2,20\nval3,30\n")
        temp_path = f.name

    try:
        sample = load_csv_column(temp_path, "col_a", n=2)
        assert len(sample) == 2
        assert all(x in ["val1", "val2", "val3"] for x in sample)
    finally:
        Path(temp_path).unlink(missing_ok=True)
