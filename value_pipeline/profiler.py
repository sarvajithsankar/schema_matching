"""
Stage 3: Statistical profiler and vectorization for column values.

================================================================================
PROFILE VECTOR LAYOUT SPECIFICATION (Length: 22 dimensions)
================================================================================
The function `profile_to_vector(profile: dict) -> list[float]` maps a
`value_profile` dictionary into a deterministic, fixed-length 22-dimensional
dense vector of floats. All values are normalized into [0.0, 1.0] or bounded
ranges [-1.0, 1.0] to ensure consistent Euclidean / cosine distance behavior.

Dim  0: is_numeric        (1.0 if dtype in ('numeric', 'int', 'float'), else 0.0)
Dim  1: is_date           (1.0 if dtype == 'date', else 0.0)
Dim  2: is_categorical    (1.0 if dtype in ('categorical', 'bool'), else 0.0)
Dim  3: is_string         (1.0 if dtype == 'string', else 0.0)
Dim  4: is_id_code        (1.0 if dtype == 'id_code', else 0.0)
Dim  5: cardinality_ratio (float in [0.0, 1.0]: distinct_values / total_values)
Dim  6: null_ratio        (float in [0.0, 1.0]: null_count / total_count)
Dim  7: length_min        (normalized via log1p(min_len) / 5.0, in [0.0, 1.0])
Dim  8: length_max        (normalized via log1p(max_len) / 5.0, in [0.0, 1.0])
Dim  9: length_mean       (normalized via log1p(mean_len) / 5.0, in [0.0, 1.0])
Dim 10: has_numeric_stats (1.0 if numeric_stats is present, else 0.0)
Dim 11: numeric_mean      (scaled via tanh(mean / 1000.0) in [-1.0, 1.0], 0 if null)
Dim 12: numeric_std       (scaled via log1p(std) / 10.0 in [0.0, 1.0], 0 if null)
Dim 13: numeric_min       (scaled via tanh(min / 1000.0) in [-1.0, 1.0], 0 if null)
Dim 14: numeric_max       (scaled via tanh(max / 1000.0) in [-1.0, 1.0], 0 if null)
Dim 15: numeric_q25       (scaled via tanh(q25 / 1000.0) in [-1.0, 1.0], 0 if null)
Dim 16: numeric_q50       (scaled via tanh(q50 / 1000.0) in [-1.0, 1.0], 0 if null)
Dim 17: numeric_q75       (scaled via tanh(q75 / 1000.0) in [-1.0, 1.0], 0 if null)
Dim 18: pattern_email     (1.0 if looks_like_email else 0.0)
Dim 19: pattern_phone     (1.0 if looks_like_phone else 0.0)
Dim 20: pattern_id_code   (1.0 if looks_like_id_code else 0.0)
Dim 21: pattern_date_str  (1.0 if looks_like_date_string else 0.0)

Total dimensions: 22.
================================================================================
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import re
import numpy as np
from dateutil import parser as date_parser

SUPPORTED_DTYPES: Tuple[str, ...] = (
    "numeric",
    "date",
    "categorical",
    "string",
    "id_code",
    "bool",
    "int",
    "float",
)

PROFILE_VECTOR_FEATURE_MAP: Dict[int, str] = {
    0: "is_numeric",
    1: "is_date",
    2: "is_categorical",
    3: "is_string",
    4: "is_id_code",
    5: "cardinality_ratio",
    6: "null_ratio",
    7: "length_min",
    8: "length_max",
    9: "length_mean",
    10: "has_numeric_stats",
    11: "numeric_mean",
    12: "numeric_std",
    13: "numeric_min",
    14: "numeric_max",
    15: "numeric_q25",
    16: "numeric_q50",
    17: "numeric_q75",
    18: "pattern_email",
    19: "pattern_phone",
    20: "pattern_id_code",
    21: "pattern_date_str",
}

assert len(PROFILE_VECTOR_FEATURE_MAP) == 22, "Profile vector layout must have exactly 22 dimensions"

EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
)

PHONE_REGEX = re.compile(
    r"^(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}$"
)

ID_CODE_REGEX = re.compile(
    r"^(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9]+([-_/][A-Za-z0-9]+)*$"
)

UUID_REGEX = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

URL_REGEX = re.compile(
    r"^(https?://|www\.)[^\s/$.?#].[^\s]*$",
    re.IGNORECASE
)

DATE_PATTERN_REGEX = re.compile(
    r"^(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4})$"
)


def _try_parse_float(val: Any) -> Optional[float]:
    """Attempt parsing value to float, returning None on failure."""
    if isinstance(val, (bool, np.bool_)):
        return None
    if isinstance(val, (int, float, np.number)):
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return None
        return float(val)
    if isinstance(val, str):
        cleaned = val.strip().replace(",", "")
        if not cleaned:
            return None
        try:
            parsed = float(cleaned)
            if np.isnan(parsed) or np.isinf(parsed):
                return None
            return parsed
        except ValueError:
            return None
    return None


def _is_bool_value(val: Any) -> bool:
    """Check if value represents a boolean."""
    if isinstance(val, (bool, np.bool_)):
        return True
    if isinstance(val, str):
        return val.strip().lower() in {"true", "false", "yes", "no", "t", "f"}
    return False


def _is_date_value(val: Any) -> bool:
    """Check if value parses as a date, guarding against plain numbers."""
    if val is None or isinstance(val, (bool, np.bool_)):
        return False
    s = str(val).strip()
    if not s:
        return False

    # Prevent pure numbers like '123' or '2024' from parsing as dates without separator
    if s.isdigit():
        return False

    # Check for presence of common date markers (- / . space or month names)
    has_date_marker = any(char in s for char in "-/.:") or any(
        m in s.lower()
        for m in [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec"
        ]
    )
    if not has_date_marker:
        return False

    try:
        date_parser.parse(s, fuzzy=False)
        return True
    except (ValueError, OverflowError, TypeError):
        return False


def _is_id_code(val: Any) -> bool:
    """Check if value matches structured ID or UUID pattern."""
    if val is None or isinstance(val, (bool, np.bool_)):
        return False
    s = str(val).strip()
    if not (2 <= len(s) <= 64):
        return False
    return bool(ID_CODE_REGEX.match(s) or UUID_REGEX.match(s))


def infer_data_type(values: list) -> str:
    """
    Infer fine-grained data type:
    'int', 'float', 'bool', 'date', 'categorical', 'string', 'id_code'.
    """
    if not values:
        return "string"

    total = len(values)

    # 1. Boolean check (>80%)
    bool_count = sum(1 for v in values if _is_bool_value(v))
    if (bool_count / total) > 0.80:
        return "bool"

    # 2. Numeric check (>80%)
    parsed_nums = [_try_parse_float(v) for v in values]
    valid_nums = [n for n in parsed_nums if n is not None]
    if (len(valid_nums) / total) > 0.80:
        # Distinguish integer vs float
        is_all_int = all(n.is_integer() for n in valid_nums)
        return "int" if is_all_int else "float"

    # 3. Date check (>80%)
    date_count = sum(1 for v in values if _is_date_value(v))
    if (date_count / total) > 0.80:
        return "date"

    # 4. ID code check (>80%)
    id_count = sum(1 for v in values if _is_id_code(v))
    if (id_count / total) > 0.80:
        return "id_code"

    # 5. Categorical check: string and cardinality_ratio < 0.05
    distinct_count = len(set(str(v) for v in values))
    cardinality_ratio = distinct_count / total
    if cardinality_ratio < 0.05:
        return "categorical"

    return "string"


def detect_dtype(values: list) -> str:
    """
    Returns one of: 'numeric', 'date', 'categorical', 'string', 'id_code'.
    Matches existing profiler API contract expected by unit tests.
    """
    fine_type = infer_data_type(values)
    if fine_type in ("int", "float"):
        return "numeric"
    if fine_type == "bool":
        return "categorical"
    return fine_type


def build_value_profile(values: list, total_count: Optional[int] = None) -> dict:
    """
    Takes a non-null value sample (output of sampler).
    Returns a value_profile dict with ALL required keys:

    {
      "dtype": str,
      "inferred_type": str,
      "cardinality_ratio": float,
      "null_ratio": float,
      "length_stats": {
          "min": int,
          "max": int,
          "mean": float
      } or None,
      "numeric_stats": {
          "mean": float,
          "std": float,
          "min": float,
          "max": float,
          "q25": float,
          "q50": float,
          "q75": float
      } or None,
      "pattern_flags": {
          "looks_like_email": bool,
          "looks_like_phone": bool,
          "looks_like_id_code": bool,
          "looks_like_date_string": bool,
          "looks_like_url": bool
      }
    }
    """
    n_sample = len(values) if values is not None else 0
    if total_count is not None and total_count > 0:
        null_count = max(0, total_count - n_sample)
        null_ratio = float(null_count / total_count)
    else:
        null_ratio = 0.0

    if n_sample == 0:
        return {
            "dtype": "string",
            "inferred_type": "string",
            "cardinality_ratio": 0.0,
            "null_ratio": 1.0 if (total_count and total_count > 0) else 0.0,
            "length_stats": {"min": 0, "max": 0, "mean": 0.0},
            "numeric_stats": None,
            "pattern_flags": {
                "looks_like_email": False,
                "looks_like_phone": False,
                "looks_like_id_code": False,
                "looks_like_date_string": False,
                "looks_like_url": False,
            },
        }

    inferred_type = infer_data_type(values)
    dtype = detect_dtype(values)
    distinct_count = len(set(str(v) for v in values))
    cardinality_ratio = float(distinct_count / n_sample)

    # Pattern flags
    str_vals = [str(v).strip() for v in values]
    email_matches = sum(1 for s in str_vals if bool(EMAIL_REGEX.match(s)))
    phone_matches = sum(1 for s in str_vals if bool(PHONE_REGEX.match(s)))
    id_matches = sum(1 for s in str_vals if _is_id_code(s))
    date_matches = sum(1 for s in str_vals if _is_date_value(s))
    url_matches = sum(1 for s in str_vals if bool(URL_REGEX.match(s)))

    pattern_flags = {
        "looks_like_email": bool((email_matches / n_sample) > 0.50),
        "looks_like_phone": bool((phone_matches / n_sample) > 0.50),
        "looks_like_id_code": bool((id_matches / n_sample) > 0.50),
        "looks_like_date_string": bool((date_matches / n_sample) > 0.50),
        "looks_like_url": bool((url_matches / n_sample) > 0.50),
    }

    # Length stats for string/id_code/categorical/bool
    if dtype in ("string", "id_code", "categorical", "bool"):
        lengths = [len(s) for s in str_vals]
        length_stats = {
            "min": int(min(lengths)),
            "max": int(max(lengths)),
            "mean": float(sum(lengths) / len(lengths)),
        }
    else:
        length_stats = None

    # Numeric stats for numeric / int / float
    if dtype in ("numeric", "int", "float"):
        num_vals = []
        for v in values:
            parsed = _try_parse_float(v)
            if parsed is not None:
                num_vals.append(parsed)

        if num_vals:
            arr = np.array(num_vals, dtype=np.float64)
            # Handle single-value or uniform arrays gracefully
            std_val = float(np.std(arr)) if len(arr) > 1 else 0.0
            numeric_stats = {
                "mean": float(np.mean(arr)),
                "std": 0.0 if np.isnan(std_val) else std_val,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "q25": float(np.percentile(arr, 25)),
                "q50": float(np.percentile(arr, 50)),
                "q75": float(np.percentile(arr, 75)),
            }
        else:
            numeric_stats = None
    else:
        numeric_stats = None

    return {
        "dtype": dtype,
        "inferred_type": inferred_type,
        "cardinality_ratio": round(cardinality_ratio, 4),
        "null_ratio": round(null_ratio, 4),
        "length_stats": length_stats,
        "numeric_stats": numeric_stats,
        "pattern_flags": pattern_flags,
    }


def profile_to_vector(profile: dict) -> list[float]:
    """
    Converts a value_profile dict into a fixed-length (22 dims) numeric vector
    for similarity computation.

    Guarantees:
    - len(vector) == 22
    - No NaN or inf values
    - All elements are floats bounded in [-1.0, 1.0] or [0.0, 1.0]
    """
    vec = [0.0] * 22

    # 1. Dtype one-hot (dims 0..4)
    dtype = profile.get("dtype", "string")
    dtype_map = {
        "numeric": 0,
        "int": 0,
        "float": 0,
        "date": 1,
        "categorical": 2,
        "bool": 2,
        "string": 3,
        "id_code": 4,
    }
    idx = dtype_map.get(dtype, 3)
    vec[idx] = 1.0

    # 2. Ratios (dims 5..6)
    c_ratio = float(profile.get("cardinality_ratio", 0.0) or 0.0)
    n_ratio = float(profile.get("null_ratio", 0.0) or 0.0)
    vec[5] = float(np.clip(c_ratio, 0.0, 1.0))
    vec[6] = float(np.clip(n_ratio, 0.0, 1.0))

    # 3. Length stats (dims 7..9)
    len_stats = profile.get("length_stats")
    if isinstance(len_stats, dict):
        min_l = float(len_stats.get("min", 0.0) or 0.0)
        max_l = float(len_stats.get("max", 0.0) or 0.0)
        mean_l = float(len_stats.get("mean", 0.0) or 0.0)
        vec[7] = float(min(1.0, math.log1p(max(0.0, min_l)) / 5.0))
        vec[8] = float(min(1.0, math.log1p(max(0.0, max_l)) / 5.0))
        vec[9] = float(min(1.0, math.log1p(max(0.0, mean_l)) / 5.0))

    # 4. Numeric stats (dims 10..17)
    num_stats = profile.get("numeric_stats")
    if isinstance(num_stats, dict):
        vec[10] = 1.0
        mean_v = float(num_stats.get("mean", 0.0) or 0.0)
        std_v = float(num_stats.get("std", 0.0) or 0.0)
        min_v = float(num_stats.get("min", 0.0) or 0.0)
        max_v = float(num_stats.get("max", 0.0) or 0.0)
        q25_v = float(num_stats.get("q25", 0.0) or 0.0)
        q50_v = float(num_stats.get("q50", 0.0) or 0.0)
        q75_v = float(num_stats.get("q75", 0.0) or 0.0)

        # Scale using tanh for bounded representation [-1.0, 1.0]
        vec[11] = float(math.tanh(mean_v / 1000.0))
        vec[12] = float(min(1.0, math.log1p(max(0.0, std_v)) / 10.0))
        vec[13] = float(math.tanh(min_v / 1000.0))
        vec[14] = float(math.tanh(max_v / 1000.0))
        vec[15] = float(math.tanh(q25_v / 1000.0))
        vec[16] = float(math.tanh(q50_v / 1000.0))
        vec[17] = float(math.tanh(q75_v / 1000.0))

    # 5. Pattern flags (dims 18..21)
    flags = profile.get("pattern_flags") or {}
    vec[18] = 1.0 if flags.get("looks_like_email") else 0.0
    vec[19] = 1.0 if flags.get("looks_like_phone") else 0.0
    vec[20] = 1.0 if flags.get("looks_like_id_code") else 0.0
    vec[21] = 1.0 if flags.get("looks_like_date_string") else 0.0

    # Ensure no NaN or inf ever
    cleaned_vec = [float(np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)) for x in vec]
    assert len(cleaned_vec) == 22, f"Profile vector length must be 22, got {len(cleaned_vec)}"
    return cleaned_vec
