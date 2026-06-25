#!/usr/bin/env python3
"""
Measure prediction error between columns target (ground truth) and generated (prediction).

Features:
- Supports CSV / XLSX / XLS input.
- Robust numeric cleaning: extracts numeric part from messy strings.
- Ignores rows with missing/invalid values in target or generated.
- Prints multiple metrics (MAE, RMSE, MAPE, sMAPE, R2, etc.).
- By default reports a fair central-band comparison on ~95% of rows,
  dropping the worst 5% (by generated error, or max error when baseline exists).
- Rows with populated leakage-feature columns are excluded from evaluation by default.
- When the baseline column is missing, only generated metrics are reported.

Usage:
    # Option 1: set defaults at script top, then run:
    python measure_error.py

    # Option 2: pass args:
    python measure_error.py --file data.csv
    python measure_error.py --file data.xlsx --target-col target --generated-col generated
    python measure_error.py --show-full-metrics
    python measure_error.py --trim-worst-fraction 0
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# Editable defaults (quick way)
# =========================
# You can edit these values directly and run: python measure_error.py
DEFAULT_FILE_PATH = "generated.csv"

# --- Task config (edit for each target) ---
DEFAULT_TARGET_COL = "esg_firma_esg-bewertung__input__elektrizitaetsverbrauch-kwh"
DEFAULT_GENERATED_COL = "generated"
DEFAULT_BASELINE_COL = ""  # optional; leave empty when no baseline column exists

LEAKAGE_FEATURES = {
    "esg_firma_environmental__elektrizitaet__wert",
    "esg_firma_environmental__elektrizitaet__einheit",
    "esg_firma_environmental__elektrizitaet__quelle",
    "esg_firma_environmental__elektrizitaet__anteil-erneuerbar-quelle",
    "esg_firma_esg-bewertung__bewertung__environmental__elektrizitaetsverbrauch-pro-kopf",
    "esg_firma_esg-bewertung__punktwert-branchendurchschnitt__elektrizitaetsverbrauch-pro-kopf",
    "esg_firma_environmental__co2-ausstoss-elektrizitaet__quelle",
    "esg_firma_environmental__elektrizitaet__anteil-erneuerbar",
    "esg_firma_environmental__co2-ausstoss-elektrizitaet__wert",
    "esg_firma_firma-esg-score-2__environmental__elektrizitaetsverbrauch",
}
DEFAULT_LEAKAGE_FEATURES = ",".join(sorted(LEAKAGE_FEATURES))
DEFAULT_TRIM_WORST_FRACTION = 0.05
DEFAULT_EXCLUDE_LEAKAGE_ROWS = True

NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_leakage_feature_list(csv_text: str) -> list[str]:
    if not csv_text or not str(csv_text).strip():
        return []
    return [part.strip() for part in str(csv_text).split(",") if part.strip()]


def column_has_leakage_value(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.notna()
    text = series.astype("string")
    return text.notna() & (text.str.strip() != "")


def drop_rows_with_leakage_values(
    df: pd.DataFrame,
    leakage_cols: list[str],
) -> tuple[pd.DataFrame, dict]:
    present = [c for c in leakage_cols if c in df.columns]
    if not present:
        return df, {
            "leakage_cols_present": [],
            "leakage_rows_dropped": 0,
        }

    leak_mask = pd.Series(False, index=df.index)
    for col in present:
        leak_mask |= column_has_leakage_value(df[col])
    dropped = int(leak_mask.sum())
    filtered = df.loc[~leak_mask].copy()
    return filtered, {
        "leakage_cols_present": present,
        "leakage_rows_dropped": dropped,
    }


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
        return pd.read_csv(file_path, low_memory=False)
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


def resolve_baseline_column(df: pd.DataFrame, baseline_col: str) -> Optional[str]:
    if not baseline_col or not str(baseline_col).strip():
        return None
    col = str(baseline_col).strip()
    if col not in df.columns:
        return None
    return col


def prepare_joint_data(
    df: pd.DataFrame,
    target_col: str,
    generated_col: str,
    baseline_col: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], dict]:
    if target_col not in df.columns:
        raise KeyError(f"Column '{target_col}' not found. Available columns: {list(df.columns)}")
    if generated_col not in df.columns:
        raise KeyError(f"Column '{generated_col}' not found. Available columns: {list(df.columns)}")

    has_baseline = baseline_col is not None
    raw_count = len(df)

    if has_baseline:
        cleaned = pd.DataFrame(
            {
                "target": df[target_col].map(extract_numeric),
                "generated": df[generated_col].map(extract_numeric),
                "baseline": df[baseline_col].map(extract_numeric),
            }
        )
        invalid_baseline = int(cleaned["baseline"].isna().sum())
        required_cols = ["target", "generated", "baseline"]
    else:
        cleaned = pd.DataFrame(
            {
                "target": df[target_col].map(extract_numeric),
                "generated": df[generated_col].map(extract_numeric),
            }
        )
        invalid_baseline = 0
        required_cols = ["target", "generated"]

    invalid_target = int(cleaned["target"].isna().sum())
    invalid_generated = int(cleaned["generated"].isna().sum())

    cleaned = cleaned.dropna(subset=required_cols)
    zero_target_rows = int((cleaned["target"] == 0).sum())
    if has_baseline:
        zero_baseline_rows = int((cleaned["baseline"] == 0).sum())
        cleaned = cleaned[(cleaned["target"] != 0) & (cleaned["baseline"] != 0)]
    else:
        zero_baseline_rows = 0
        cleaned = cleaned[cleaned["target"] != 0]

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
        "zero_target_rows": zero_target_rows,
        "zero_baseline_rows": zero_baseline_rows,
        "has_baseline": has_baseline,
        "baseline_col": baseline_col,
    }

    y_base = cleaned["baseline"].to_numpy(dtype=float) if has_baseline else None
    return (
        cleaned["target"].to_numpy(dtype=float),
        cleaned["generated"].to_numpy(dtype=float),
        y_base,
        info,
    )


def fair_outlier_trim_mask(
    y_true: np.ndarray,
    y_gen: np.ndarray,
    y_base: Optional[np.ndarray],
    trim_worst_fraction: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Drop the worst rows using the same mask for both methods when baseline exists.

    Score per row:
    - with baseline: max(|generated-target|, |baseline-target|)
    - without baseline: |generated-target|
    """
    n = len(y_true)
    keep_all = np.ones(n, dtype=bool)
    info: Dict[str, float] = {
        "trim_worst_fraction": float(trim_worst_fraction),
        "trimmed_rows": 0.0,
        "kept_rows": float(n),
        "trim_score_threshold": float("nan"),
    }
    if trim_worst_fraction <= 0 or n == 0:
        return keep_all, info

    if not (0.0 < trim_worst_fraction < 0.5):
        raise ValueError("--trim-worst-fraction must be in (0, 0.5).")

    err_gen = np.abs(y_gen - y_true)
    if y_base is None:
        score = err_gen
    else:
        err_base = np.abs(y_base - y_true)
        score = np.maximum(err_gen, err_base)
    k_drop = int(math.ceil(n * trim_worst_fraction))
    if k_drop <= 0:
        return keep_all, info
    if k_drop >= n:
        return np.zeros(n, dtype=bool), {
            **info,
            "trimmed_rows": float(n),
            "kept_rows": 0.0,
            "trim_score_threshold": float(np.max(score)),
        }

    order = np.argsort(score, kind="mergesort")
    keep_idx = order[: n - k_drop]
    keep = np.zeros(n, dtype=bool)
    keep[keep_idx] = True
    worst_kept_score = float(np.max(score[keep_idx]))
    worst_dropped_score = float(np.min(score[~keep])) if k_drop > 0 else float("nan")
    info.update(
        {
            "trimmed_rows": float(k_drop),
            "kept_rows": float(keep.sum()),
            "trim_score_threshold": worst_kept_score,
            "min_dropped_score": worst_dropped_score,
            "max_dropped_score": float(np.max(score[~keep])),
        }
    )
    return keep, info


def format_metrics_table(
    metrics_gen: dict,
    metrics_base: Optional[dict],
    generated_col: str,
    baseline_col: Optional[str],
) -> pd.DataFrame:
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
    gen_header = f"{generated_col} (generated)"
    base_header = f"{baseline_col} (baseline)" if baseline_col else None
    include_baseline = metrics_base is not None and base_header is not None
    for key in metric_order:
        v_gen = metrics_gen[key]
        if key == "count_used":
            gen_text = str(int(v_gen))
        else:
            gen_text = fmt(v_gen)
        row = {
            "metric": metric_name_map[key],
            gen_header: gen_text,
        }
        if include_baseline:
            v_base = metrics_base[key]
            if key == "count_used":
                base_text = str(int(v_base))
            else:
                base_text = fmt(v_base)
            row[base_header] = base_text
        rows.append(row)
    return pd.DataFrame(rows)


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
        help=(
            "Optional baseline column for comparison. Leave empty or omit the column "
            "in the file to evaluate generated predictions only."
        ),
    )
    parser.add_argument(
        "--leakage-features",
        default=DEFAULT_LEAKAGE_FEATURES,
        help=(
            "Comma-separated leakage columns. When --exclude-leakage-rows is enabled, "
            "rows with any non-empty value in these columns are dropped before metrics."
        ),
    )
    parser.add_argument(
        "--exclude-leakage-rows",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_EXCLUDE_LEAKAGE_ROWS,
        help=(
            "Drop rows where any leakage feature column has a value before computing metrics "
            f"(default: {DEFAULT_EXCLUDE_LEAKAGE_ROWS})."
        ),
    )
    parser.add_argument(
        "--trim-worst-fraction",
        type=float,
        default=DEFAULT_TRIM_WORST_FRACTION,
        help=(
            "Drop the worst f fraction of rows (by max error of generated/baseline) "
            "before reporting primary metrics. Set 0 to evaluate all rows. "
            f"(default: {DEFAULT_TRIM_WORST_FRACTION})"
        ),
    )
    parser.add_argument(
        "--show-full-metrics",
        action="store_true",
        help="Also print metrics on all valid rows (in addition to trimmed primary report).",
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
    if not (0.0 <= args.trim_worst_fraction < 0.5):
        print("[ERROR] --trim-worst-fraction must be in [0, 0.5).")
        return 1

    try:
        df = load_table(file_path)
        leakage_cols = parse_leakage_feature_list(args.leakage_features)
        leakage_info = {
            "leakage_cols_present": [],
            "leakage_rows_dropped": 0,
        }
        if args.exclude_leakage_rows and leakage_cols:
            df, leakage_info = drop_rows_with_leakage_values(df, leakage_cols)
            if leakage_info["leakage_rows_dropped"] > 0:
                print(
                    f"[INFO] Excluded {leakage_info['leakage_rows_dropped']} row(s) with leakage "
                    f"feature values ({len(leakage_info['leakage_cols_present'])} column(s) checked)."
                )
            elif leakage_info["leakage_cols_present"]:
                print(
                    "[INFO] Leakage feature columns present but no populated rows were found; "
                    "kept all rows for evaluation."
                )
        baseline_col_resolved = resolve_baseline_column(df, args.baseline_col)
        y_true, y_pred_gen, y_pred_base, info = prepare_joint_data(
            df, args.target_col, args.generated_col, baseline_col_resolved
        )
        has_baseline = bool(info["has_baseline"])
        info.update(leakage_info)
        info["exclude_leakage_rows"] = bool(args.exclude_leakage_rows)
        info["leakage_feature_roots"] = len(leakage_cols)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("=== Data Cleaning Summary ===")
    print(f"File: {file_path}")
    print(f"Total rows: {info['raw_rows']}")
    if has_baseline:
        print(f"Used rows (shared by generated & baseline): {info['used_rows']}")
    else:
        print(f"Used rows (target & generated): {info['used_rows']}")
        print("Baseline column: not used (missing or not configured)")
    print(f"Dropped rows: {info['dropped_rows']}")
    if has_baseline:
        print(
            "Rows with invalid values (target/generated/baseline): "
            f"{info['invalid_target_rows']}/"
            f"{info['invalid_generated_rows']}/"
            f"{info['invalid_baseline_rows']}"
        )
        print(
            "Rows filtered by zero value (target/baseline): "
            f"{info['zero_target_rows']}/{info['zero_baseline_rows']}"
        )
    else:
        print(
            "Rows with invalid values (target/generated): "
            f"{info['invalid_target_rows']}/{info['invalid_generated_rows']}"
        )
        print(f"Rows filtered by zero value (target): {info['zero_target_rows']}")
    if info.get("exclude_leakage_rows"):
        present = info.get("leakage_cols_present") or []
        print(
            "Leakage row filter: "
            f"{info.get('leakage_rows_dropped', 0)} row(s) dropped, "
            f"{len(present)} leakage column(s) present in file"
        )
    print()

    if (
        has_baseline
        and y_pred_base is not None
        and np.allclose(y_pred_gen, y_pred_base, rtol=0.0, atol=1e-9, equal_nan=True)
    ):
        print(
            "[NOTE] generated and baseline predictions are identical on all evaluated rows.\n"
            "       Training likely used alpha=0 and/or fallback to baseline (no correction applied).\n"
            "       Re-run train_and_generate.py after tuning changes, or check Training Summary for\n"
            "       'Best alpha' and 'Fallback to baseline correction'."
        )
        print()

    y_true_eval = y_true
    y_gen_eval = y_pred_gen
    y_base_eval = y_pred_base
    trim_info: Dict[str, float] | None = None

    if args.trim_worst_fraction > 0:
        keep_mask, trim_info = fair_outlier_trim_mask(
            y_true, y_pred_gen, y_pred_base, args.trim_worst_fraction
        )
        if int(trim_info["kept_rows"]) == 0:
            print("[WARN] Fair trim removed all rows; falling back to all valid rows.")
        else:
            y_true_eval = y_true[keep_mask]
            y_gen_eval = y_pred_gen[keep_mask]
            y_base_eval = y_pred_base[keep_mask]

    metrics_gen = compute_metrics(y_true_eval, y_gen_eval)
    metrics_base = compute_metrics(y_true_eval, y_base_eval) if has_baseline and y_base_eval is not None else None
    table_df = format_metrics_table(
        metrics_gen,
        metrics_base,
        args.generated_col,
        baseline_col_resolved,
    )

    if trim_info is not None and int(trim_info["kept_rows"]) > 0:
        kept_pct = 100.0 * (1.0 - args.trim_worst_fraction)
        print(
            f"=== Primary Metrics (fair trim, central ~{kept_pct:.0f}% rows) ==="
        )
        if has_baseline:
            print(
                "Trim rule: drop the worst "
                f"{args.trim_worst_fraction:.0%} rows by "
                "max(|generated-target|, |baseline-target|); "
                "same rows removed for both methods (likely noisy tail)."
            )
        else:
            print(
                "Trim rule: drop the worst "
                f"{args.trim_worst_fraction:.0%} rows by "
                "|generated-target| (likely noisy tail)."
            )
        print(
            f"Rows kept/trimmed: {int(trim_info['kept_rows'])}/"
            f"{int(trim_info['trimmed_rows'])} "
            f"(score threshold kept<={trim_info['trim_score_threshold']:.6f}, "
            f"dropped>={trim_info.get('min_dropped_score', float('nan')):.6f})"
        )
    else:
        print("=== Error Metrics Comparison (target=true, all valid rows) ===")
    print(table_df.to_string(index=False))

    if args.show_full_metrics and args.trim_worst_fraction > 0 and trim_info is not None:
        if int(trim_info["kept_rows"]) > 0 and int(trim_info["kept_rows"]) < len(y_true):
            metrics_gen_all = compute_metrics(y_true, y_pred_gen)
            metrics_base_all = (
                compute_metrics(y_true, y_pred_base) if has_baseline and y_pred_base is not None else None
            )
            table_all_df = format_metrics_table(
                metrics_gen_all,
                metrics_base_all,
                args.generated_col,
                baseline_col_resolved,
            )
            print()
            print("=== Full Metrics (all valid rows, includes noisy tail) ===")
            print(table_all_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
