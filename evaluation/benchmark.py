"""
Benchmarking harness for schema matching.
Loads benchmark pairs from synthetic and Valentine datasets,
executes Adaptive Fusion against 3 baselines (Name-Only, Value-Only, Fixed-Fusion),
prints detailed per-scenario and overall comparative tables,
and exports results to evaluation/results.csv.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import csv
import json
import numpy as np

from evaluation.metrics import compute_metrics, compute_macro_metrics
from evaluation.baselines import (
    baseline_name_only,
    baseline_value_only,
    baseline_fixed_fusion,
    adaptive_matcher,
)
from data.valentine.loader import load_all_valentine_pairs

DEFAULT_SYNTHETIC_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"
DEFAULT_RESULTS_CSV = Path(__file__).resolve().parent / "results.csv"


def load_synthetic_pairs(synthetic_dir: Optional[str] = None) -> List[dict]:
    """
    Load synthetic schema pairs from data/synthetic/.
    Excludes standalone ground-truth JSON files.
    """
    target_dir = Path(synthetic_dir) if synthetic_dir else DEFAULT_SYNTHETIC_DIR
    if not target_dir.exists():
        return []

    pairs = []
    # Filter only pair_*.json that are not *_ground_truth.json
    files = sorted([
        f for f in target_dir.glob("pair_*.json")
        if not f.name.endswith("_ground_truth.json")
    ])

    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            if "ground_truth" in data:
                data["ground_truth"] = [tuple(item) for item in data["ground_truth"]]
            if "file_name" not in data:
                data["file_name"] = f.name
            pairs.append(data)

    return pairs


def load_all_benchmark_pairs(
    synthetic_dir: Optional[str] = None,
    valentine_dir: Optional[str] = None
) -> List[dict]:
    """
    Load all benchmark pairs combining synthetic and Valentine datasets.
    """
    all_pairs: List[dict] = []

    synthetic_pairs = load_synthetic_pairs(synthetic_dir)
    all_pairs.extend(synthetic_pairs)

    val_pairs = load_all_valentine_pairs(valentine_dir)
    all_pairs.extend(val_pairs)

    return all_pairs


def run_benchmark(
    schema_pairs: List[dict],
    matcher_fn: Callable,
    **matcher_kwargs
) -> dict:
    """
    Runs matcher_fn on every schema pair, collects metrics per pair,
    and returns aggregate mean precision / recall / F1 across all pairs.
    """
    if not schema_pairs:
        return {
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "mean_f1": 0.0,
            "pair_results": []
        }

    pair_results = []
    metrics_list = []

    for pair in schema_pairs:
        profiles_a = pair.get("profiles_a", [])
        profiles_b = pair.get("profiles_b", [])
        gt = pair.get("ground_truth", [])

        matches = matcher_fn(profiles_a, profiles_b, **matcher_kwargs)
        metrics = compute_metrics(matches, gt)

        metrics_list.append(metrics)
        pair_results.append({
            "pair_id": pair.get("pair_id", pair.get("file_name", "unknown")),
            "description": pair.get("description", ""),
            "metrics": metrics,
            "matches": [m.to_dict() if hasattr(m, "to_dict") else m for m in matches]
        })

    macro = compute_macro_metrics(metrics_list)

    return {
        "mean_precision": macro["mean_precision"],
        "mean_recall": macro["mean_recall"],
        "mean_f1": macro["mean_f1"],
        "pair_results": pair_results
    }


def print_comparison_table(results: Dict[str, dict], schema_pairs: List[dict]):
    """
    Print an ASCII comparison table showing per-scenario and overall results.
    """
    print("\n" + "=" * 88)
    print(" " * 24 + "SCHEMA MATCHING BENCHMARK RESULTS")
    print("=" * 88)

    # 1. Per-Scenario breakdown
    for i, pair in enumerate(schema_pairs, 1):
        pid = pair.get("pair_id", f"pair_{i}")
        desc = pair.get("description", "")
        print(f"\nScenario: {pid} - {desc}")
        print(f"{'Matcher':<30} | {'Precision':<15} | {'Recall':<15} | {'F1':<15}")
        print("-" * 88)
        for m_name, res in results.items():
            # Find metrics for this pair
            p_res = next((p for p in res["pair_results"] if p["pair_id"] == pid), None)
            if p_res:
                m = p_res["metrics"]
                print(f"{m_name:<30} | {m['precision']:<15.4f} | {m['recall']:<15.4f} | {m['f1']:<15.4f}")

    # 2. Overall Summary Table
    print("\n" + "=" * 88)
    print(" " * 28 + "OVERALL MACRO SUMMARY")
    print("=" * 88)
    print(f"{'Matcher Strategy':<30} | {'Mean Precision':<15} | {'Mean Recall':<15} | {'Mean F1':<15}")
    print("-" * 88)
    for name, res in results.items():
        p = f"{res['mean_precision']:.4f}"
        r = f"{res['mean_recall']:.4f}"
        f = f"{res['mean_f1']:.4f}"
        print(f"{name:<30} | {p:<15} | {r:<15} | {f:<15}")
    print("=" * 88 + "\n")


def save_results_to_csv(
    results: Dict[str, dict],
    schema_pairs: List[dict],
    csv_path: Optional[Path] = None
) -> Path:
    """
    Save per-scenario and overall macro benchmark results to CSV.
    """
    output_path = csv_path if csv_path else DEFAULT_RESULTS_CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Scenario", "Description", "Matcher", "Precision", "Recall", "F1", "TP", "FP", "FN"])

        for pair in schema_pairs:
            pid = pair.get("pair_id", "")
            desc = pair.get("description", "")
            for m_name, res in results.items():
                p_res = next((p for p in res["pair_results"] if p["pair_id"] == pid), None)
                if p_res:
                    m = p_res["metrics"]
                    writer.writerow([pid, desc, m_name, m["precision"], m["recall"], m["f1"], m["tp"], m["fp"], m["fn"]])

        # Summary rows
        for m_name, res in results.items():
            writer.writerow([
                "OVERALL_MEAN",
                "Macro average across all pairs",
                m_name,
                res["mean_precision"],
                res["mean_recall"],
                res["mean_f1"],
                "", "", ""
            ])

    print(f"[OK] Results saved to {output_path}")
    return output_path


def main():
    pairs = load_all_benchmark_pairs()
    if not pairs:
        print("[ERROR] No benchmark pairs found. Run python generate_synthetic.py first.")
        return

    print(f"Loaded {len(pairs)} benchmark schema pairs.")

    matchers = {
        "1. Name-Only Baseline": baseline_name_only,
        "2. Value-Only Baseline": baseline_value_only,
        "3. Fixed-Fusion (alpha=0.5)": baseline_fixed_fusion,
        "4. Adaptive Fusion": adaptive_matcher,
    }

    benchmark_results: Dict[str, dict] = {}
    for name, fn in matchers.items():
        benchmark_results[name] = run_benchmark(pairs, fn)

    print_comparison_table(benchmark_results, pairs)
    save_results_to_csv(benchmark_results, pairs)


if __name__ == "__main__":
    main()
