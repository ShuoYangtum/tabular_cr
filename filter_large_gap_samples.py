#!/usr/bin/env python3
"""
Filter rows where |wasser_berechnet - esg-bewertung__input__wasserverbrauch-m3| > threshold
from both train and test files, then save cleaned files and print removal ratios.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# Editable defaults
# =========================
TRAIN_INPUT = "../modified/train.csv"
TEST_INPUT = "../modified/test.csv"
TRAIN_OUTPUT = "../modified/train_clean.csv"
TEST_OUTPUT = "../modified/test_clean.csv"

BASELINE_COL = "wasser_berechnet"
TARGET_COL = "esg-bewertung__input__wasserverbrauch-m3"
THRESHOLD = 1e9


NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
MISSING_MARKERS = {"", "nan", "none", "null", "na", "n/a", "-", "--", "unknown"}


def extract_numeric(value) -> float:
    if value is None:
        return np.nan

    if isinstance(value, (int, float, np.number)):
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return np.nan
        return v

    s = str(value).strip()
    if not s:
        return np.nan

    s = s.replace(",", "")
    s = re.sub(r"\s+", "", s)
    if s.lower() in MISSING_MARKERS:
        return np.nan

    m = NUMERIC_PATTERN.search(s)
    if not m:
        return np.nan

    try:
        v = float(m.group())
        if np.isnan(v) or np.isinf(v):
            return np.nan
        return v
    except (ValueError, OverflowError):
        return np.nan


def clean_one_file(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    if BASELINE_COL not in df.columns:
        raise KeyError(f"Column '{BASELINE_COL}' not found in {input_path}")
    if TARGET_COL not in df.columns:
        raise KeyError(f"Column '{TARGET_COL}' not found in {input_path}")

    total_rows = len(df)
    baseline_num = df[BASELINE_COL].map(extract_numeric)
    target_num = df[TARGET_COL].map(extract_numeric)
    valid_pair = baseline_num.notna() & target_num.notna()

    # Only filter rows where both columns are valid numeric and the gap exceeds threshold.
    to_remove = valid_pair & ((baseline_num - target_num).abs() > THRESHOLD)
    cleaned_df = df.loc[~to_remove].copy()

    removed_rows = int(to_remove.sum())
    removed_ratio = (removed_rows / total_rows) if total_rows > 0 else 0.0
    valid_rows = int(valid_pair.sum())
    removed_ratio_in_valid = (removed_rows / valid_rows) if valid_rows > 0 else 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"=== {input_path.name} ===")
    print(f"Total rows: {total_rows}")
    print(f"Valid numeric pairs ({BASELINE_COL}, {TARGET_COL}): {valid_rows}")
    print(f"Removed rows (|gap| > {THRESHOLD:.0f}): {removed_rows}")
    print(f"Removed ratio (all rows): {removed_ratio:.4%}")
    print(f"Removed ratio (valid pairs only): {removed_ratio_in_valid:.4%}")
    print(f"Saved: {output_path}")
    print()


def main() -> None:
    train_input = Path(TRAIN_INPUT)
    test_input = Path(TEST_INPUT)
    train_output = Path(TRAIN_OUTPUT)
    test_output = Path(TEST_OUTPUT)

    clean_one_file(train_input, train_output)
    clean_one_file(test_input, test_output)


if __name__ == "__main__":
    main()
