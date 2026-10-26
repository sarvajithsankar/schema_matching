"""Unit tests for evaluation.metrics."""

import pytest
from evaluation.metrics import compute_metrics, compute_macro_metrics
from matching.matcher import SchemaMatch


def test_compute_metrics_perfect():
    predicted = [
        {"column_a": "A.c1", "column_b": "B.c1", "score": 0.95, "accepted": True},
        {"column_a": "A.c2", "column_b": "B.c2", "score": 0.88, "accepted": True},
    ]
    ground_truth = [("A.c1", "B.c1"), ("A.c2", "B.c2")]

    metrics = compute_metrics(predicted, ground_truth)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0


def test_compute_metrics_partial_manual():
    # Ground truth has 3 pairs: (A1, B1), (A2, B2), (A3, B3)
    # Predicted has:
    # (A1, B1) -> TP
    # (A2, B4) -> FP
    # (A3, B3) -> not predicted => FN
    # (A4, B2) accepted=False -> ignored
    predicted = [
        {"column_a": "A1", "column_b": "B1", "score": 0.9, "accepted": True},
        {"column_a": "A2", "column_b": "B4", "score": 0.7, "accepted": True},
        {"column_a": "A4", "column_b": "B2", "score": 0.2, "accepted": False},
    ]
    ground_truth = [("A1", "B1"), ("A2", "B2"), ("A3", "B3")]

    metrics = compute_metrics(predicted, ground_truth)

    # TP = 1, FP = 1, FN = 2
    # Precision = 1 / (1 + 1) = 0.5
    # Recall = 1 / (1 + 2) = 1/3 ≈ 0.3333
    # F1 = 2 * (0.5 * 0.3333) / (0.5 + 0.3333) = 0.4
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 2
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == pytest.approx(0.3333, abs=1e-4)
    assert metrics["f1"] == pytest.approx(0.4000, abs=1e-4)


def test_compute_metrics_zero_division():
    # Empty predictions and empty ground truth
    m0 = compute_metrics([], [])
    assert m0["precision"] == 0.0
    assert m0["recall"] == 0.0
    assert m0["f1"] == 0.0

    # Predictions made but ground truth is empty (all FP)
    m_no_gt = compute_metrics([("A", "B")], [])
    assert m_no_gt["precision"] == 0.0
    assert m_no_gt["recall"] == 0.0
    assert m_no_gt["f1"] == 0.0

    # Ground truth exists but no predictions (all FN)
    m_no_pred = compute_metrics([], [("A", "B")])
    assert m_no_pred["precision"] == 0.0
    assert m_no_pred["recall"] == 0.0
    assert m_no_pred["f1"] == 0.0


def test_compute_metrics_with_schema_matches():
    matches = [
        SchemaMatch("A.id", "B.id", 0.95, accepted=True),
        SchemaMatch("A.extra", "B.wrong", 0.80, accepted=True),
    ]
    gt = [("A.id", "B.id"), ("A.name", "B.name")]

    metrics = compute_metrics(matches, gt)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_compute_macro_metrics():
    list_of_metrics = [
        {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        {"precision": 0.5, "recall": 0.5, "f1": 0.5},
    ]
    macro = compute_macro_metrics(list_of_metrics)
    assert macro["mean_precision"] == 0.75
    assert macro["mean_recall"] == 0.75
    assert macro["mean_f1"] == 0.75

    empty_macro = compute_macro_metrics([])
    assert empty_macro["mean_f1"] == 0.0
