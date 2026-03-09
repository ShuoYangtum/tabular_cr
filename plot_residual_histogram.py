#!/usr/bin/env python3
"""
Plot residual histogram from a table and save as an image.

Residual definition:
    residual = generated - target
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================
# Editable defaults (quick way)
# =========================
DEFAULT_FILE_PATH = "generated.csv"
DEFAULT_TARGET_COL = "esg-bewertung__input__wasserverbrauch-m3"
DEFAULT_GENERATED_COL = "generated"
DEFAULT_OUTPUT_IMAGE = "residual_histogram.png"
DEFAULT_BINS = 50


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


def fmt(v: float) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "NaN"
    return f"{v:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot residual histogram (generated - target).")
    parser.add_argument("--file", default=DEFAULT_FILE_PATH, help="Input table path (.csv/.xlsx/.xls)")
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL, help="True value column name")
    parser.add_argument("--generated-col", default=DEFAULT_GENERATED_COL, help="Predicted value column name")
    parser.add_argument("--output-image", default=DEFAULT_OUTPUT_IMAGE, help="Output image file path")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS, help="Histogram bins")
    args = parser.parse_args()

    if not args.file:
        print("[ERROR] No input file provided. Set DEFAULT_FILE_PATH or pass --file.")
        return 1
    if args.bins <= 0:
        print("[ERROR] --bins must be positive.")
        return 1

    table_path = Path(args.file)
    output_path = Path(args.output_image)
    if not table_path.exists():
        print(f"[ERROR] File not found: {table_path}")
        return 1

    try:
        df = load_table(table_path)
    except Exception as exc:
        print(f"[ERROR] Failed to read file: {exc}")
        return 1

    df.columns = [str(c).strip() for c in df.columns]
    if args.target_col not in df.columns:
        print(f"[ERROR] target column '{args.target_col}' not found.")
        print(f"Available columns: {list(df.columns)}")
        return 1
    if args.generated_col not in df.columns:
        print(f"[ERROR] generated column '{args.generated_col}' not found.")
        print(f"Available columns: {list(df.columns)}")
        return 1

    target = df[args.target_col].map(extract_numeric)
    generated = df[args.generated_col].map(extract_numeric)
    valid = target.notna() & generated.notna()
    if not valid.any():
        print("[ERROR] No valid paired rows after cleaning.")
        return 1

    residual = (generated[valid] - target[valid]).to_numpy(dtype=float)

    plt.figure(figsize=(10, 6))
    plt.hist(residual, bins=args.bins, edgecolor="black", alpha=0.75)
    plt.axvline(0.0, color="red", linestyle="--", linewidth=1.2, label="Zero error")
    plt.title("Residual Histogram (generated - target)")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
    except Exception as exc:
        print(f"[ERROR] Failed to save image: {exc}")
        return 1
    finally:
        plt.close()

    print("=== Residual Histogram Summary ===")
    print(f"Input file: {table_path}")
    print(f"Output image: {output_path}")
    print(f"Target column: {args.target_col}")
    print(f"Generated column: {args.generated_col}")
    print(f"Total rows: {len(df)}")
    print(f"Valid paired rows used: {int(valid.sum())}")
    print(f"Dropped rows: {int((~valid).sum())}")
    print(f"Mean residual: {fmt(float(np.mean(residual)))}")
    print(f"Std residual: {fmt(float(np.std(residual)))}")
    print(f"Median residual: {fmt(float(np.median(residual)))}")
    print(f"MAE residual: {fmt(float(np.mean(np.abs(residual))))}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
