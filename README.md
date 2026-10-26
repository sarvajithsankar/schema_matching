# Schema Matching Using Adaptive Fusion

A Python project for matching columns across schemas by combining column-name similarity with value-profile similarity. The matcher adapts the relative weight of these signals and uses globally optimal one-to-one assignment to produce schema correspondences.

## Overview

The project evaluates whether columns from two different schemas represent the same underlying data. It combines:

- **Name similarity** based on normalized column names and cosine similarity between embeddings.
- **Value profiling** based on inferred data types, cardinality, null ratios, length statistics, numeric statistics, and common patterns such as emails, phone numbers, IDs, and dates.
- **Adaptive fusion** that changes the name/value weighting based on name quality, similarity, and disagreement between signals.
- **Global matching** using the Hungarian algorithm from SciPy.

## Repository structure

```text
data/
  synthetic/       Generated schema-pair benchmark data
  test_schemas/    Schema fixtures
  valentine/       Placeholder for external Valentine benchmark datasets

evaluation/
  baselines.py     Baseline name-only, value-only, and fixed-fusion matchers
  benchmark.py     Benchmark loading and comparison harness
  metrics.py       Precision, recall, and F1 calculations

matching/
  matcher.py       Similarity matrix construction and Hungarian matching

person_a/
  fusion.py        Adaptive name/value fusion
  name_similarity.py
                   Cosine-based name similarity helpers
  normalize_names.py
                   Column-name normalization utilities

shared/
  contracts.py     Shared profile contracts
  column_profile_schema.json
                   Column profile schema definition

value_pipeline/
  profiler.py      Column value profiling and 22-dimensional vectorization
  sampler.py       Value sampling utilities
  similarity.py    Value-profile similarity calculations

tests/             Unit tests for the pipeline and matcher
generate_synthetic.py
                   Synthetic benchmark-pair generator
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Generate benchmark data

The synthetic generator creates five schema-pair scenarios, including exact matches, renamed columns, conflicting data types, mixed matchability, and a customer/client example:

```bash
python generate_synthetic.py
```

Generated pairs are written to `data/synthetic/`.

## Run the benchmark

```bash
python -m evaluation.benchmark
```

The benchmark compares Adaptive Fusion against Name-Only, Value-Only, and Fixed-Fusion baselines across all benchmark scenarios, reports per-scenario and mean precision, recall, and F1 scores, and writes outputs to `evaluation/results.csv`.

## Results

Across the 5 synthetic benchmark scenarios, the matchers achieve the following macro performance:

| Matcher Strategy | Mean Precision | Mean Recall | Mean F1 |
| :--- | :--- | :--- | :--- |
| **1. Name-Only Baseline** | 0.6667 | 0.5917 | 0.5538 |
| **2. Value-Only Baseline** | 0.9000 | 1.0000 | 0.9333 |
| **3. Fixed-Fusion (alpha=0.5)** | 0.8667 | 1.0000 | 0.9000 |
| **4. Adaptive Fusion** | 0.8667 | 1.0000 | 0.9000 |

### Per-Scenario Breakdown

- **Pair 1 (Exact Match)**: All matchers achieve P=1.0, R=1.0, F1=1.0.
- **Pair 2 (Renamed Columns, Same Values)**: Value-Only, Fixed-Fusion, and Adaptive Fusion achieve F1=1.0, while Name-Only fails completely (F1=0.0).
- **Pair 3 (Similar Names, Conflicting Types)**: Name-Only (F1=0.50), Fixed-Fusion (F1=0.50), and Adaptive Fusion (F1=0.50) are penalized by identical column names with mismatched value profiles (`age` numeric vs date). Value-Only achieves higher precision (P=0.50, F1=0.6667) because it relies solely on value divergence.
- **Pair 4 (Mixed Matchability)**: Value-Only, Fixed-Fusion, and Adaptive Fusion achieve F1=1.0, whereas Name-Only misses renamed columns (R=0.3333, F1=0.50).
- **Pair 5 (Customer vs Client Schemas)**: Value-Only, Fixed-Fusion, and Adaptive Fusion achieve F1=1.0, whereas Name-Only achieves F1=0.7692.

## Valentine Benchmark Datasets

External datasets from the Valentine schema matching benchmark (e.g., ChFin, OpenData, Magellan) can be evaluated by placing dataset directories in `data/valentine/<dataset_name>/`. Each folder should contain:
- `source.csv` (or `table1.csv`)
- `target.csv` (or `table2.csv`)
- `ground_truth.csv` (with `src_col,tgt_col,is_match` or `source,target`)

If no Valentine datasets are present, the benchmark harness skips them gracefully.

## Run tests

```bash
pytest
```

## Main matching flow

A column profile contains identifiers, normalized name tokens, optional name embeddings, sampled values, and a structured `value_profile`. The value profiler converts each profile into a fixed 22-dimensional vector. Name and value similarities are combined into a similarity matrix, and `matching.matcher.match_schemas()` applies the Hungarian algorithm to select one-to-one matches above the configured confidence threshold.

## Dependencies

The project uses NumPy, SciPy, pandas, scikit-learn, python-dateutil, and pytest. See [`requirements.txt`](requirements.txt) for the version constraints.
