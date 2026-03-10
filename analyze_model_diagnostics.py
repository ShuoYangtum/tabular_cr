#!/usr/bin/env python3
"""
Privacy-safe diagnostics for baseline/generated vs target.

Outputs aggregate statistics only (JSON + CSV), no raw rows.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# =========================
# Editable defaults
# =========================
DEFAULT_FILE = "generated.csv"
DEFAULT_TARGET_COL = "esg-bewertung__input__wasserverbrauch-m3"
DEFAULT_BASELINE_COL = "wasser_berechnet"
DEFAULT_GENERATED_COL = "generated"
DEFAULT_OUTPUT_JSON = "diagnostics_summary.json"
DEFAULT_SEGMENT_CSV = "diagnostics_by_baseline_bin.csv"
DEFAULT_BINS = 10


NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
MISSING_STRINGS = {"", "nan", "none", "null", "na", "n/a", "-", "--", "unknown"}
SAFE_MAX = np.finfo(np.float64).max * 0.99
EPS = 1e-12


def extract_numeric(value) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        v = float(value)
        if math.isnan(v) or math.isinf(v) or abs(v) > SAFE_MAX:
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
        if math.isnan(v) or math.isinf(v) or abs(v) > SAFE_MAX:
            return np.nan
        return v
    except Exception:
        return np.nan


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    bias = float(np.mean(err))
    medae = float(np.median(np.abs(err)))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot)
    return {"mae": mae, "rmse": rmse, "r2": r2, "bias": bias, "median_ae": medae}


def quantiles(arr: np.ndarray) -> Dict[str, float]:
    qs = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    vals = np.quantile(arr, qs)
    return {f"q{int(q*100):02d}": float(v) for q, v in zip(qs, vals)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze baseline/generated diagnostics.")
    parser.add_argument("--file", default=DEFAULT_FILE, help="Input table path")
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL, help="Target column")
    parser.add_argument("--baseline-col", default=DEFAULT_BASELINE_COL, help="Baseline column")
    parser.add_argument("--generated-col", default=DEFAULT_GENERATED_COL, help="Generated column")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON, help="Summary JSON output")
    parser.add_argument("--segment-csv", default=DEFAULT_SEGMENT_CSV, help="Segment CSV output")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS, help="Baseline quantile bins")
    args = parser.parse_args()

    if args.bins < 2:
        raise ValueError("--bins must be >= 2")

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError("Use csv/xlsx/xls")

    for col in [args.target_col, args.baseline_col, args.generated_col]:
        if col not in df.columns:
            raise KeyError(f"Column not found: {col}")

    target = df[args.target_col].map(extract_numeric)
    baseline = df[args.baseline_col].map(extract_numeric)
    generated = df[args.generated_col].map(extract_numeric)

    valid_base = target.notna() & baseline.notna()
    valid_gen = target.notna() & generated.notna()

    y_b = target[valid_base].to_numpy(dtype=float)
    p_b = baseline[valid_base].to_numpy(dtype=float)
    y_g = target[valid_gen].to_numpy(dtype=float)
    p_g = generated[valid_gen].to_numpy(dtype=float)

    baseline_m = metrics(y_b, p_b) if len(y_b) > 0 else {}
    generated_m = metrics(y_g, p_g) if len(y_g) > 0 else {}

    # ratio / correction diagnostics on generated-valid rows
    ratio_bg = np.full_like(y_g, np.nan, dtype=float)
    nonzero_base = np.abs(p_b[: len(y_g)]) > EPS if len(y_g) <= len(p_b) else np.abs(p_g) > EPS
    if len(y_g) > 0:
        ratio = y_g / np.where(np.abs(p_g) > EPS, p_g, np.nan)
        ratio_bg = ratio
    ratio_bg = ratio_bg[np.isfinite(ratio_bg)]

    # Segment by baseline quantiles (on rows where target/baseline/generated all valid)
    valid_all = target.notna() & baseline.notna() & generated.notna()
    seg_df = pd.DataFrame(
        {
            "target": target[valid_all].to_numpy(dtype=float),
            "baseline": baseline[valid_all].to_numpy(dtype=float),
            "generated": generated[valid_all].to_numpy(dtype=float),
        }
    )

    if len(seg_df) > 0:
        seg_df["bin"] = pd.qcut(
            seg_df["baseline"],
            q=min(args.bins, seg_df["baseline"].nunique()),
            labels=False,
            duplicates="drop",
        )
        by_rows: List[Dict[str, float]] = []
        for b, g in seg_df.groupby("bin", dropna=True):
            y = g["target"].to_numpy(dtype=float)
            pb = g["baseline"].to_numpy(dtype=float)
            pg = g["generated"].to_numpy(dtype=float)
            mb = metrics(y, pb)
            mg = metrics(y, pg)
            by_rows.append(
                {
                    "bin": int(b),
                    "rows": int(len(g)),
                    "baseline_mae": mb["mae"],
                    "generated_mae": mg["mae"],
                    "mae_gain": mb["mae"] - mg["mae"],
                    "baseline_rmse": mb["rmse"],
                    "generated_rmse": mg["rmse"],
                    "rmse_gain": mb["rmse"] - mg["rmse"],
                }
            )
        by_bin_df = pd.DataFrame(by_rows).sort_values("bin")
    else:
        by_bin_df = pd.DataFrame(
            columns=[
                "bin",
                "rows",
                "baseline_mae",
                "generated_mae",
                "mae_gain",
                "baseline_rmse",
                "generated_rmse",
                "rmse_gain",
            ]
        )

    summary = {
        "file": str(path),
        "rows_total": int(len(df)),
        "rows_valid_baseline": int(valid_base.sum()),
        "rows_valid_generated": int(valid_gen.sum()),
        "rows_invalid_generated": int((~valid_gen).sum()),
        "baseline_metrics": baseline_m,
        "generated_metrics": generated_m,
        "delta_metrics_generated_minus_baseline": (
            {
                "mae_delta": generated_m.get("mae", np.nan) - baseline_m.get("mae", np.nan),
                "rmse_delta": generated_m.get("rmse", np.nan) - baseline_m.get("rmse", np.nan),
                "r2_delta": generated_m.get("r2", np.nan) - baseline_m.get("r2", np.nan),
            }
            if baseline_m and generated_m
            else {}
        ),
        "target_quantiles": quantiles(target[target.notna()].to_numpy(dtype=float))
        if target.notna().any()
        else {},
        "baseline_quantiles": quantiles(baseline[baseline.notna()].to_numpy(dtype=float))
        if baseline.notna().any()
        else {},
        "generated_quantiles": quantiles(generated[generated.notna()].to_numpy(dtype=float))
        if generated.notna().any()
        else {},
        "ratio_target_over_generated_quantiles": quantiles(ratio_bg) if len(ratio_bg) > 0 else {},
    }

    out_json = Path(args.output_json)
    out_csv = Path(args.segment_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    by_bin_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Summary JSON: {out_json}")
    print(f"Segment CSV: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
