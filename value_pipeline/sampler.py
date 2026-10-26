"""
Stage 1: Value sampling module.
Pulls representative non-null samples from column values or CSV sources.
Handles nulls, empty columns, huge columns, and mixed types deterministically.
"""

from pathlib import Path
from typing import Any, List, Optional, Sequence, Union
import random
import numpy as np
import pandas as pd


def _is_null(val: Any) -> bool:
    """
    Check if a value is considered null, missing, or empty.
    Safely handles numbers, booleans, strings, and pandas NA/NaT.
    """
    if val is None:
        return True
    # Fast path for booleans (bool is a subclass of int in Python)
    if isinstance(val, (bool, np.bool_)):
        return False
    # Check for pandas/numpy NaN, NA, NaT
    try:
        if pd.isna(val):
            return True
    except (TypeError, ValueError):
        pass

    if isinstance(val, (float, np.floating)) and np.isnan(val):
        return True

    if isinstance(val, str):
        cleaned = val.strip().lower()
        if cleaned in {"", "null", "none", "nan", "na", "<na>", "n/a", "undefined"}:
            return True

    return False


def sample_column(
    values: Optional[Sequence[Any]],
    n: int = 500,
    random_state: Optional[int] = 42
) -> List[Any]:
    """
    Given a sequence of raw column values (which may contain None/NaN/empty/mixed types),
    return up to n non-null values sampled without replacement deterministically.

    - If values is None or empty, returns [].
    - If non-null count <= n, returns all non-null values.
    - If huge sequence, samples efficiently with fixed seed.
    """
    if values is None:
        return []

    # If pandas Series or numpy array, convert to iterable
    if isinstance(values, pd.Series):
        # Drop known nulls fast via pandas
        valid_series = values.dropna()
        raw_list = valid_series.tolist()
    elif isinstance(values, np.ndarray):
        raw_list = values.tolist()
    else:
        raw_list = list(values)

    if not raw_list:
        return []

    # Filter nulls and empty representations
    non_null = [v for v in raw_list if not _is_null(v)]
    total_non_null = len(non_null)

    if total_non_null == 0:
        return []

    if total_non_null <= n:
        return non_null

    rng = random.Random(random_state) if random_state is not None else random
    return rng.sample(non_null, n)


def load_csv_column(
    filepath: Union[str, Path],
    column_name: str,
    n: int = 500,
    random_state: Optional[int] = 42,
    chunksize: int = 10000
) -> List[Any]:
    """
    Load a column from a CSV file and return a sample of up to n non-null values.
    Supports large CSV files efficiently via chunked reading when appropriate.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    # For reasonably sized files, direct read is fastest
    try:
        df = pd.read_csv(filepath, usecols=[column_name], low_memory=False)
        raw_values = df[column_name].tolist()
        return sample_column(raw_values, n=n, random_state=random_state)
    except Exception:
        # Fallback to chunked reading for massive files
        collected: List[Any] = []
        for chunk in pd.read_csv(filepath, usecols=[column_name], chunksize=chunksize, low_memory=False):
            vals = chunk[column_name].tolist()
            non_null_chunk = [v for v in vals if not _is_null(v)]
            collected.extend(non_null_chunk)
            if len(collected) >= n * 5:
                break
        return sample_column(collected, n=n, random_state=random_state)
