"""
Valentine benchmark dataset loader.
Converts Valentine-format dataset pairs (source.csv, target.csv, ground_truth.csv/json)
into the standard ColumnProfile pair format for benchmarking.
If no Valentine datasets are found, skips gracefully with an informative message.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import pandas as pd

from person_a.normalize_names import normalize_column_name
from shared.contracts import ColumnProfile
from value_pipeline.profiler import build_value_profile
from value_pipeline.sampler import sample_column

DEFAULT_VALENTINE_DIR = Path(__file__).resolve().parent


def load_valentine_pair(dataset_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load a single Valentine dataset directory containing:
    - source.csv (or table1.csv)
    - target.csv (or table2.csv)
    - ground_truth.csv (or ground_truth.json)
    """
    path = Path(dataset_dir)
    if not path.is_dir():
        return None

    # Identify source CSV
    src_file = path / "source.csv"
    if not src_file.exists():
        src_file = path / "table1.csv"

    # Identify target CSV
    tgt_file = path / "target.csv"
    if not tgt_file.exists():
        tgt_file = path / "table2.csv"

    if not src_file.exists() or not tgt_file.exists():
        return None

    df_src = pd.read_csv(src_file)
    df_tgt = pd.read_csv(tgt_file)

    schema_a_name = path.name + "_source"
    schema_b_name = path.name + "_target"

    profiles_a: List[Dict[str, Any]] = []
    for col in df_src.columns:
        vals = df_src[col].tolist()
        sample = sample_column(vals, n=500, random_state=42)
        vp = build_value_profile(sample, total_count=len(vals))
        tokens = normalize_column_name(str(col))
        col_prof = ColumnProfile(
            column_id=f"{schema_a_name}.{col}",
            schema_name=schema_a_name,
            raw_name=str(col),
            normalized_tokens=tokens,
            sampled_values=sample[:20],
            name_embedding=[],
            value_profile=vp,
        )
        profiles_a.append(col_prof.to_dict())

    profiles_b: List[Dict[str, Any]] = []
    for col in df_tgt.columns:
        vals = df_tgt[col].tolist()
        sample = sample_column(vals, n=500, random_state=42)
        vp = build_value_profile(sample, total_count=len(vals))
        tokens = normalize_column_name(str(col))
        col_prof = ColumnProfile(
            column_id=f"{schema_b_name}.{col}",
            schema_name=schema_b_name,
            raw_name=str(col),
            normalized_tokens=tokens,
            sampled_values=sample[:20],
            name_embedding=[],
            value_profile=vp,
        )
        profiles_b.append(col_prof.to_dict())

    # Load ground truth
    gt_pairs: List[Tuple[str, str]] = []
    gt_csv = path / "ground_truth.csv"
    gt_json = path / "ground_truth.json"

    if gt_csv.exists():
        gt_df = pd.read_csv(gt_csv)
        # Check column names: src_col/tgt_col or source/target
        src_c = next((c for c in gt_df.columns if "src" in c.lower() or "source" in c.lower()), gt_df.columns[0])
        tgt_c = next((c for c in gt_df.columns if "tgt" in c.lower() or "target" in c.lower()), gt_df.columns[1])
        match_col = next((c for c in gt_df.columns if "match" in c.lower()), None)

        for _, row in gt_df.iterrows():
            if match_col is not None:
                if str(row[match_col]).strip() not in {"1", "True", "true", "1.0"}:
                    continue
            src_val = f"{schema_a_name}.{row[src_c]}"
            tgt_val = f"{schema_b_name}.{row[tgt_c]}"
            gt_pairs.append((src_val, tgt_val))

    elif gt_json.exists():
        with open(gt_json, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
            if isinstance(gt_data, dict):
                for s, t in gt_data.items():
                    gt_pairs.append((f"{schema_a_name}.{s}", f"{schema_b_name}.{t}"))
            elif isinstance(gt_data, list):
                for item in gt_data:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        gt_pairs.append((f"{schema_a_name}.{item[0]}", f"{schema_b_name}.{item[1]}"))

    return {
        "pair_id": path.name,
        "description": f"Valentine Dataset: {path.name}",
        "profiles_a": profiles_a,
        "profiles_b": profiles_b,
        "ground_truth": gt_pairs,
    }


def load_all_valentine_pairs(valentine_dir: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """
    Search valentine_dir for subdirectories containing Valentine benchmark pairs.
    If none found, prints an informative skip message and returns [].
    """
    target_dir = Path(valentine_dir) if valentine_dir else DEFAULT_VALENTINE_DIR
    if not target_dir.exists():
        print(f"[INFO] Valentine directory {target_dir} does not exist. Skipping.")
        return []

    subdirs = [p for p in target_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    pairs: List[Dict[str, Any]] = []

    for d in subdirs:
        pair = load_valentine_pair(d)
        if pair is not None:
            pairs.append(pair)

    if not pairs:
        print("[INFO] No Valentine benchmark dataset files present in data/valentine/ (skipped gracefully).")
        print("       See README.md for instructions on downloading Valentine datasets.")

    return pairs
