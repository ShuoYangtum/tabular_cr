#!/usr/bin/env python3
"""
Measure prediction error between columns target (ground truth) and generated (prediction).

Features:
- Supports CSV / XLSX / XLS input.
- Robust numeric cleaning: extracts numeric part from messy strings.
- Ignores rows with missing/invalid values in target or generated.
- Prints multiple metrics (MAE, RMSE, MAPE, sMAPE, R2, etc.).

Usage:
    # Option 1: set defaults at script top, then run:
    python measure_error.py

    # Option 2: pass args:
    python measure_error.py --file data.csv
    python measure_error.py --file data.xlsx --target-col target --generated-col generated
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# =========================
# Editable defaults (quick way)
# =========================
# You can edit these values directly and run: python measure_error.py
DEFAULT_FILE_PATH = "generated.csv"
DEFAULT_TARGET_COL = "esg_firma_esg-bewertung__input__wasserverbrauch-m3"
DEFAULT_GENERATED_COL = "generated" # esg_firma_wasser_berechnet
DEFAULT_BASELINE_COL = "esg_firma_wasser_berechnet"

NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def extract_numeric(value) -> float:
    """
    Convert potentially messy values to float.
    Examples:
      "12.3kg" -> 12.3
      "about -7" -> -7.0
      "1,234.56" -> 1234.56
      "" / invalid -> NaN
    """
    if value is None:
        return np.nan

    if isinstance(value, (int, float, np.number)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return np.nan
        return float(value)

    s = str(value).strip()
    if not s:
        return np.nan

    # Normalize common separators and whitespace.
    s = s.replace(",", "")
    s = re.sub(r"\s+", "", s)

    # Handle explicit missing markers.
    if s.lower() in {"nan", "none", "null", "na", "n/a", "-"}:
        return np.nan

    match = NUMERIC_PATTERN.search(s)
    if not match:
        return np.nan

    try:
        num = float(match.group())
        if math.isinf(num) or math.isnan(num):
            return np.nan
        return num
    except (ValueError, OverflowError):
        return np.nan


def load_table(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported file type: {suffix}. Use .csv/.xlsx/.xls")


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Exclude rows where true value is 0 to avoid division by zero.
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def safe_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100)


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # R2 undefined when y_true variance is zero.
    if len(y_true) < 2:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


def safe_mdape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return float(np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def safe_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def trimmed_mean(arr: np.ndarray, trim_ratio: float) -> float:
    if len(arr) == 0:
        return np.nan
    if not (0 <= trim_ratio < 0.5):
        raise ValueError("trim_ratio must be in [0, 0.5).")
    lo = int(len(arr) * trim_ratio)
    hi = len(arr) - lo
    if hi <= lo:
        return np.nan
    arr_sorted = np.sort(arr)
    return float(np.mean(arr_sorted[lo:hi]))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    abs_err = np.abs(err)

    mae = float(np.mean(abs_err))
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    medae = float(np.median(abs_err))
    bias = float(np.mean(err))  # positive means over-prediction
    mape = safe_mape(y_true, y_pred)
    mdape = safe_mdape(y_true, y_pred)
    smape = safe_smape(y_true, y_pred)
    wape = safe_wape(y_true, y_pred)
    r2 = safe_r2(y_true, y_pred)
    p90_ae = float(np.quantile(abs_err, 0.90))
    p95_ae = float(np.quantile(abs_err, 0.95))
    trimmed_mae_90 = trimmed_mean(abs_err, 0.05)  # drop top/bottom 5%
    trimmed_rmse_90 = float(np.sqrt(trimmed_mean(err**2, 0.05)))

    target_median = float(np.median(np.abs(y_true)))
    nmae_by_target_median = float(mae / target_median) if target_median > 0 else np.nan

    return {
        "count_used": int(len(y_true)),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "median_ae": medae,
        "p90_ae": p90_ae,
        "p95_ae": p95_ae,
        "trimmed_mae_90": trimmed_mae_90,
        "trimmed_rmse_90": trimmed_rmse_90,
        "bias_mean_error": bias,
        "mape_percent": mape,
        "mdape_percent": mdape,
        "smape_percent": smape,
        "wape_percent": wape,
        "nmae_by_target_median": nmae_by_target_median,
        "r2": r2,
    }


def prepare_joint_data(
    df: pd.DataFrame,
    target_col: str,
    generated_col: str,
    baseline_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if target_col not in df.columns:
        raise KeyError(f"Column '{target_col}' not found. Available columns: {list(df.columns)}")
    if generated_col not in df.columns:
        raise KeyError(f"Column '{generated_col}' not found. Available columns: {list(df.columns)}")
    if baseline_col not in df.columns:
        raise KeyError(f"Column '{baseline_col}' not found. Available columns: {list(df.columns)}")

    raw_count = len(df)

    cleaned = pd.DataFrame(
        {
            "target": df[target_col].map(extract_numeric),
            "generated": df[generated_col].map(extract_numeric),
            "baseline": df[baseline_col].map(extract_numeric),
        }
    )

    invalid_target = int(cleaned["target"].isna().sum())
    invalid_generated = int(cleaned["generated"].isna().sum())
    invalid_baseline = int(cleaned["baseline"].isna().sum())

    cleaned = cleaned.dropna(subset=["target", "generated", "baseline"])
    used_count = len(cleaned)
    dropped_count = raw_count - used_count

    if used_count == 0:
        raise ValueError("No valid paired numeric rows after cleaning. Check your data format.")

    info = {
        "raw_rows": raw_count,
        "used_rows": used_count,
        "dropped_rows": dropped_count,
        "invalid_target_rows": invalid_target,
        "invalid_generated_rows": invalid_generated,
        "invalid_baseline_rows": invalid_baseline,
    }

    return (
        cleaned["target"].to_numpy(dtype=float),
        cleaned["generated"].to_numpy(dtype=float),
        cleaned["baseline"].to_numpy(dtype=float),
        info,
    )


def fmt(v: float) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "NaN"
    return f"{v:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate errors between columns target (true) and generated (pred)."
    )
    parser.add_argument(
        "--file",
        default=DEFAULT_FILE_PATH,
        help="Input table file path (.csv/.xlsx/.xls). If omitted, use DEFAULT_FILE_PATH in script.",
    )
    parser.add_argument(
        "--target-col",
        default=DEFAULT_TARGET_COL,
        help=f"Column name for true values (default: {DEFAULT_TARGET_COL})",
    )
    parser.add_argument(
        "--generated-col",
        default=DEFAULT_GENERATED_COL,
        help=f"Column name for predicted values (default: {DEFAULT_GENERATED_COL})",
    )
    parser.add_argument(
        "--baseline-col",
        default=DEFAULT_BASELINE_COL,
        help=f"Column name for baseline values (default: {DEFAULT_BASELINE_COL})",
    )
    args = parser.parse_args()

    if not args.file:
        print(
            "[ERROR] No input file provided. Set DEFAULT_FILE_PATH at top of script "
            "or pass --file."
        )
        return 1

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return 1

    try:
        df = load_table(file_path)
        y_true, y_pred_gen, y_pred_base, info = prepare_joint_data(
            df, args.target_col, args.generated_col, args.baseline_col
        )
        metrics_gen = compute_metrics(y_true, y_pred_gen)
        metrics_base = compute_metrics(y_true, y_pred_base)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("=== Data Cleaning Summary ===")
    print(f"File: {file_path}")
    print(f"Total rows: {info['raw_rows']}")
    print(f"Used rows (shared by generated & baseline): {info['used_rows']}")
    print(f"Dropped rows: {info['dropped_rows']}")
    print(
        "Rows with invalid values (target/generated/baseline): "
        f"{info['invalid_target_rows']}/"
        f"{info['invalid_generated_rows']}/"
        f"{info['invalid_baseline_rows']}"
    )
    print()

    metric_order = [
        "count_used",
        "mae",
        "mse",
        "rmse",
        "median_ae",
        "p90_ae",
        "p95_ae",
        "trimmed_mae_90",
        "trimmed_rmse_90",
        "bias_mean_error",
        "mape_percent",
        "mdape_percent",
        "smape_percent",
        "wape_percent",
        "nmae_by_target_median",
        "r2",
    ]
    metric_name_map = {
        "count_used": "count_used",
        "mae": "MAE",
        "mse": "MSE",
        "rmse": "RMSE",
        "median_ae": "MedianAE",
        "p90_ae": "P90AE",
        "p95_ae": "P95AE",
        "trimmed_mae_90": "TrimmedMAE90",
        "trimmed_rmse_90": "TrimmedRMSE90",
        "bias_mean_error": "Bias(mean err)",
        "mape_percent": "MAPE(%)",
        "mdape_percent": "MdAPE(%)",
        "smape_percent": "sMAPE(%)",
        "wape_percent": "WAPE(%)",
        "nmae_by_target_median": "nMAE/med(|y|)",
        "r2": "R2",
    }

    rows = []
    for key in metric_order:
        v_gen = metrics_gen[key]
        v_base = metrics_base[key]
        if key == "count_used":
            gen_text = str(int(v_gen))
            base_text = str(int(v_base))
        else:
            gen_text = fmt(v_gen)
            base_text = fmt(v_base)
        rows.append(
            {
                "metric": metric_name_map[key],
                f"{args.generated_col} (generated)": gen_text,
                f"{args.baseline_col} (baseline)": base_text,
            }
        )

    table_df = pd.DataFrame(rows)
    print("=== Error Metrics Comparison (target=true) ===")
    print(table_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
