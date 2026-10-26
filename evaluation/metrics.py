"""
Evaluation metrics for schema matching.
Computes Precision, Recall, and F1 score against ground truth mappings,
safely handling zero division and macro averaging across pairs.
"""

from typing import Any, Dict, List, Sequence, Set, Tuple
import numpy as np


def compute_metrics(
    predicted_matches: Sequence[Any],
    ground_truth: Sequence[Tuple[str, str]]
) -> Dict[str, float]:
    """
    Computes precision, recall, and F1 score for schema matching predictions.

    Parameters
    ----------
    predicted_matches : Sequence[Any]
        Output matches from match_schemas. Each item can be a SchemaMatch tuple
        (column_a, column_b, score), a dictionary, or a 2-tuple (column_a, column_b).
        Only matches with accepted=True are evaluated.
    ground_truth : Sequence[Tuple[str, str]]
        List or set of correct (column_a_id, column_b_id) pairs.

    Returns
    -------
    dict
        {
          "precision": float,
          "recall": float,
          "f1": float,
          "tp": int,
          "fp": int,
          "fn": int
        }
    """
    pred_pairs: Set[Tuple[str, str]] = set()

    for m in predicted_matches:
        if isinstance(m, dict):
            if m.get("accepted", True):
                pred_pairs.add((str(m["column_a"]), str(m["column_b"])))
        elif hasattr(m, "accepted"):
            if m.accepted:
                pred_pairs.add((str(m.column_a), str(m.column_b)))
        elif isinstance(m, (tuple, list)):
            if len(m) >= 2:
                pred_pairs.add((str(m[0]), str(m[1])))

    gt_pairs: Set[Tuple[str, str]] = {
        (str(pair[0]), str(pair[1])) for pair in ground_truth
    }

    tp = len(pred_pairs & gt_pairs)
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (
        float((2 * precision * recall) / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_macro_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Compute macro-averaged precision, recall, and F1 across multiple schema pairs.
    Handles empty lists gracefully.
    """
    if not metrics_list:
        return {"mean_precision": 0.0, "mean_recall": 0.0, "mean_f1": 0.0}

    precisions = [m["precision"] for m in metrics_list]
    recalls = [m["recall"] for m in metrics_list]
    f1s = [m["f1"] for m in metrics_list]

    return {
        "mean_precision": round(float(np.mean(precisions)), 4),
        "mean_recall": round(float(np.mean(recalls)), 4),
        "mean_f1": round(float(np.mean(f1s)), 4),
    }
