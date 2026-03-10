#!/usr/bin/env python3

import argparse
import math
import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# =========================
# Editable defaults
# =========================
DEFAULT_FILE_PATH = "../modified/test.csv"
DEFAULT_TARGET_COL = "esg-bewertung__input__wasserverbrauch-m3"
DEFAULT_GENERATED_COL = "wasser_berechnet"
NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
MISSING_STRINGS = {"", "nan", "none", "null", "na", "n/a", "-", "--", "unknown"}
FLOAT64_SAFE_MAX = np.finfo(np.float64).max * 0.99

def extract_numeric(value) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        v = float(value)
        if math.isnan(v) or math.isinf(v) or abs(v) > FLOAT64_SAFE_MAX:
            return np.nan
        return v
    s = str(value).strip()
    if not s:
        return np.nan
    s = s.replace(",", "")
    s = re.sub(r"\s+", "", s)
    if s.lower() in MISSING_STRINGS:
        return np.nan
    m = NUMERIC_PATTERN.search(s)
    if not m:
        return np.nan
    try:
        v = float(m.group())
        if math.isnan(v) or math.isinf(v) or abs(v) > FLOAT64_SAFE_MAX:
            return np.nan
        return v
    except (ValueError, OverflowError):
        return np.nan

def load_table(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported file type: {suffix}. Use .csv/.xlsx/.xls")

def main() -> int:
    parser = argparse.ArgumentParser(description="Calculate correlation factors for residuals and all numeric columns.")
    parser.add_argument("--file", default=DEFAULT_FILE_PATH, help="Input table path (.csv/.xlsx/.xls)")
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL, help="True value column name")
    parser.add_argument("--generated-col", default=DEFAULT_GENERATED_COL, help="Predicted value column name")
    args = parser.parse_args()

    table_path = Path(args.file)
    if not table_path.exists():
        print(f"[ERROR] File not found: {table_path}")
        return 1

    try:
        df = load_table(table_path)
    except Exception as exc:
        print(f"[ERROR] Failed to read file: {exc}")
        return 1

    df.columns = [str(c).strip() for c in df.columns]
    if args.target_col not in df.columns or args.generated_col not in df.columns:
        print(f"[ERROR] Required columns not found. Available columns: {list(df.columns)}")
        return 1

    target = df[args.target_col].map(extract_numeric)
    generated = df[args.generated_col].map(extract_numeric)
    valid = target.notna() & generated.notna()

    if not valid.any():
        print("[ERROR] No valid paired rows after cleaning.")
        return 1

    residual = (generated[valid] - target[valid]).to_numpy(dtype=float)
    small_residual_mask = residual < 1e-10
    small_residual_indices = np.where(small_residual_mask)[0]

    if len(small_residual_indices) == 0:
        print("[INFO] No samples with residuals smaller than -1e-10.")
        return 0

    print("=== Top-5 Correlated Columns ===")
    correlations = {}
    for col in df.columns:
        if col in {args.target_col, args.generated_col}:
            continue  # Skip the target and generated columns
        feature_values = df.loc[valid, col].map(extract_numeric).to_numpy(dtype=float)
        feature_subset = feature_values[small_residual_indices]
        residual_subset = residual[small_residual_indices]

        # Skip columns with insufficient valid data
        if len(feature_subset) > 0 and not np.all(np.isnan(feature_subset)):
            try:
                # Remove NaN values before calculating correlation
                valid_indices = ~np.isnan(feature_subset) & ~np.isnan(residual_subset)
                feature_subset = feature_subset[valid_indices]
                residual_subset = residual_subset[valid_indices]

                if len(feature_subset) > 0:  # Ensure valid data remains
                    correlation = np.corrcoef(residual_subset, feature_subset)[0, 1]
                    correlations[col] = abs(correlation)  # Use absolute value for sorting
            except Exception:
                correlations[col] = None  # Handle cases where correlation fails

    # Filter out columns with None correlation values and sort by absolute correlation
    sorted_correlations = {
        k: v for k, v in sorted(correlations.items(), key=lambda item: item[1], reverse=True) if v is not None
    }

    # Print top-5 correlated columns
    for i, (col, corr) in enumerate(sorted_correlations.items()):
        if i >= 5:
            break
        print(f"{i+1}. Column: {col}, Absolute Correlation: {corr:.6f}")

    print("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
