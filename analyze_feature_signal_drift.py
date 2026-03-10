#!/usr/bin/env python3
"""
Privacy-safe feature signal + drift analysis.

Outputs per-feature aggregated diagnostics (no raw data rows):
- missing/unique/type parse stats
- correlation with target and correction target on train
- train-test drift proxy (PSI-like score for numeric, TV distance for categorical)
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# =========================
# Editable defaults
# =========================
DEFAULT_TRAIN_FILE = "../modified/train_3_clean.csv"
DEFAULT_TEST_FILE = "../modified/test_3_clean.csv"
DEFAULT_TARGET_COL = "esg_firma_esg-bewertung__input__wasserverbrauch-m3"
DEFAULT_BASELINE_COL = "esg_firma_wasser_berechnet"
DEFAULT_OUTPUT_CSV = "feature_signal_drift_report.csv"
DEFAULT_NUM_BINS = 10
DEFAULT_MIN_NON_NULL = 100


NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
MISSING_STRINGS = {"", "nan", "none", "null", "na", "n/a", "-", "--", "unknown"}
EPS = 1e-12


def extract_numeric(value) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        v = float(value)
        if math.isnan(v) or math.isinf(v):
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
        if math.isnan(v) or math.isinf(v):
            return np.nan
        return v
    except Exception:
        return np.nan


def psi_numeric(train: np.ndarray, test: np.ndarray, bins: int) -> float:
    if len(train) == 0 or len(test) == 0:
        return np.nan
    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(train, qs))
    if len(edges) < 3:
        return 0.0
    train_hist, _ = np.histogram(train, bins=edges)
    test_hist, _ = np.histogram(test, bins=edges)
    train_p = train_hist / max(train_hist.sum(), 1)
    test_p = test_hist / max(test_hist.sum(), 1)
    train_p = np.clip(train_p, 1e-6, 1.0)
    test_p = np.clip(test_p, 1e-6, 1.0)
    return float(np.sum((test_p - train_p) * np.log(test_p / train_p)))


def tv_categorical(train: pd.Series, test: pd.Series, top_k: int = 30) -> float:
    tr = train.astype(str).fillna("__nan__")
    te = test.astype(str).fillna("__nan__")
    top = tr.value_counts().head(top_k).index.tolist()
    tr_freq = tr.value_counts(normalize=True)
    te_freq = te.value_counts(normalize=True)
    keys = set(top) | set(te.value_counts().head(top_k).index.tolist())
    dist = 0.0
    for k in keys:
        dist += abs(float(tr_freq.get(k, 0.0)) - float(te_freq.get(k, 0.0)))
    return 0.5 * dist


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze feature signal and train-test drift.")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE, help="Train file")
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE, help="Test file")
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL, help="Target column")
    parser.add_argument("--baseline-col", default=DEFAULT_BASELINE_COL, help="Baseline column")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="Output CSV")
    parser.add_argument("--num-bins", type=int, default=DEFAULT_NUM_BINS, help="Bins for numeric drift")
    parser.add_argument("--min-non-null", type=int, default=DEFAULT_MIN_NON_NULL, help="Min non-null for correlation")
    args = parser.parse_args()

    train_path = Path(args.train_file)
    test_path = Path(args.test_file)
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not test_path.exists():
        raise FileNotFoundError(test_path)

    train = pd.read_csv(train_path, low_memory=False) if train_path.suffix.lower() == ".csv" else pd.read_excel(train_path)
    test = pd.read_csv(test_path, low_memory=False) if test_path.suffix.lower() == ".csv" else pd.read_excel(test_path)
    train.columns = [str(c).strip() for c in train.columns]
    test.columns = [str(c).strip() for c in test.columns]

    for c in [args.target_col, args.baseline_col]:
        if c not in train.columns:
            raise KeyError(f"Train missing column: {c}")

    target = train[args.target_col].map(extract_numeric)
    baseline = train[args.baseline_col].map(extract_numeric)
    corr_target = np.sign(target) * np.log1p(np.abs(target)) - np.sign(baseline) * np.log1p(np.abs(baseline))

    feature_cols = [c for c in train.columns if c not in {args.target_col, args.baseline_col}]
    rows: List[Dict[str, float]] = []
    for col in feature_cols:
        tr_col = train[col]
        te_col = test[col] if col in test.columns else pd.Series([np.nan] * len(test))

        tr_non_null = int(tr_col.notna().sum())
        te_non_null = int(te_col.notna().sum())
        tr_non_null_ratio = float(tr_col.notna().mean())
        te_non_null_ratio = float(te_col.notna().mean())
        nunique = int(tr_col.nunique(dropna=True))
        uniq_ratio = float(nunique / max(tr_non_null, 1))

        tr_num = tr_col.map(extract_numeric)
        te_num = te_col.map(extract_numeric)
        tr_num_ratio = float(tr_num.notna().mean())
        te_num_ratio = float(te_num.notna().mean())

        inferred_type = "categorical"
        if tr_num_ratio >= 0.8:
            inferred_type = "numeric"

        corr_target_valid = corr_target.notna()
        num_corr = np.nan
        if inferred_type == "numeric":
            mask = tr_num.notna() & corr_target_valid
            if int(mask.sum()) >= args.min_non_null:
                num_corr = float(pd.Series(tr_num[mask]).corr(pd.Series(corr_target[mask]), method="spearman"))
            drift = psi_numeric(
                tr_num[tr_num.notna()].to_numpy(dtype=float),
                te_num[te_num.notna()].to_numpy(dtype=float),
                bins=args.num_bins,
            )
            drift_kind = "psi_numeric"
        else:
            drift = tv_categorical(tr_col, te_col)
            drift_kind = "tv_categorical"

        id_like_flag = int(uniq_ratio > 0.95 and tr_non_null > 200)

        rows.append(
            {
                "feature": col,
                "inferred_type": inferred_type,
                "train_non_null_ratio": tr_non_null_ratio,
                "test_non_null_ratio": te_non_null_ratio,
                "train_numeric_parse_ratio": tr_num_ratio,
                "test_numeric_parse_ratio": te_num_ratio,
                "train_unique_ratio_non_null": uniq_ratio,
                "id_like_flag": id_like_flag,
                "spearman_with_log_correction": num_corr,
                "drift_metric_name": drift_kind,
                "drift_metric_value": drift,
            }
        )

    report = pd.DataFrame(rows)
    report = report.sort_values(
        ["id_like_flag", "drift_metric_value", "spearman_with_log_correction"],
        ascending=[False, False, False],
        na_position="last",
    )
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Output: {out}")
    print(f"Features analyzed: {len(report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
