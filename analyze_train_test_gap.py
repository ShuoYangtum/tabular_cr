#!/usr/bin/env python3
"""
Privacy-safe train/test gap diagnostics for tabular regression correction.

Outputs ONLY aggregated statistics (JSON + CSV). No raw rows, no cell values.

Designed to support improvement decisions when raw data cannot be shared.
Run after train_and_generate.py and optionally measure_error.py.

Usage:
    python analyze_train_test_gap.py
    python analyze_train_test_gap.py --train-file train_clean.csv --test-file test_clean.csv
    python analyze_train_test_gap.py --generated-file generated.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =========================
# Editable defaults
# =========================
DEFAULT_TRAIN_FILE = "train_clean.csv"
DEFAULT_TEST_FILE = "test_clean.csv"
DEFAULT_GENERATED_FILE = "generated.csv"
DEFAULT_TARGET_COL = "esg_firma_esg-bewertung__input__wasserverbrauch-m3"
DEFAULT_BASELINE_COL = "esg_firma_wasser_berechnet"
DEFAULT_GENERATED_COL = "generated"
DEFAULT_OUTPUT_DIR = "gap_diagnostics"
DEFAULT_DRIFT_TOP_K = 25
DEFAULT_BASELINE_BINS = 10
DEFAULT_EPS = 1e-12

NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
MISSING_STRINGS = {"", "nan", "none", "null", "na", "n/a", "-", "--", "unknown"}


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
    except (ValueError, OverflowError):
        return np.nan


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def quantile_dict(arr: np.ndarray, prefix: str = "") -> Dict[str, float]:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    qs = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    out: Dict[str, float] = {}
    for q in qs:
        key = f"{prefix}q{int(round(q * 100)):02d}" if prefix else f"q{int(round(q * 100)):02d}"
        out[key] = float(np.quantile(arr, q))
    out[f"{prefix}mean" if prefix else "mean"] = float(np.mean(arr))
    out[f"{prefix}std" if prefix else "std"] = float(np.std(arr))
    out[f"{prefix}count" if prefix else "count"] = float(arr.size)
    return out


def psi_from_train_edges(train: np.ndarray, test: np.ndarray, bins: int = 10) -> float:
    train = train[np.isfinite(train)]
    test = test[np.isfinite(test)]
    if train.size == 0 or test.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(train, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 3:
        return 0.0
    tr_hist, _ = np.histogram(train, bins=edges)
    te_hist, _ = np.histogram(test, bins=edges)
    tr_p = np.clip(tr_hist / max(tr_hist.sum(), 1), 1e-6, 1.0)
    te_p = np.clip(te_hist / max(te_hist.sum(), 1), 1e-6, 1.0)
    return float(np.sum((te_p - tr_p) * np.log(te_p / tr_p)))


def tv_categorical(train: pd.Series, test: pd.Series, top_k: int = 30) -> float:
    tr = train.astype(str).fillna("__nan__")
    te = test.astype(str).fillna("__nan__")
    keys = set(tr.value_counts().head(top_k).index) | set(te.value_counts().head(top_k).index)
    tr_freq = tr.value_counts(normalize=True)
    te_freq = te.value_counts(normalize=True)
    return float(0.5 * sum(abs(float(tr_freq.get(k, 0.0)) - float(te_freq.get(k, 0.0))) for k in keys))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    trimmed = abs_err.copy()
    if trimmed.size >= 20:
        lo = int(trimmed.size * 0.05)
        hi = trimmed.size - lo
        trimmed_mae90 = float(np.mean(np.sort(trimmed)[lo:hi]))
    else:
        trimmed_mae90 = mae
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "median_ae": float(np.median(abs_err)),
        "bias": float(np.mean(err)),
        "trimmed_mae90": trimmed_mae90,
    }


def error_tail_concentration(y_true: np.ndarray, y_pred: np.ndarray) -> List[Dict[str, float]]:
    abs_err = np.abs(y_pred - y_true)
    total = float(abs_err.sum())
    if total <= 0 or abs_err.size == 0:
        return []
    order = np.argsort(-abs_err)
    rows: List[Dict[str, float]] = []
    for top_pct in (0.01, 0.05, 0.10, 0.20):
        k = max(int(math.ceil(abs_err.size * top_pct)), 1)
        idx = order[:k]
        share = float(abs_err[idx].sum() / total)
        rows.append(
            {
                "top_fraction": top_pct,
                "top_rows": int(k),
                "mae_share": share,
                "mean_abs_err_in_top": float(abs_err[idx].mean()),
                "mean_target_in_top": float(y_true[idx].mean()),
                "mean_pred_in_top": float(y_pred[idx].mean()),
            }
        )
    return rows


def valid_label_mask(target: pd.Series, baseline: pd.Series) -> pd.Series:
    t = target.map(extract_numeric)
    b = baseline.map(extract_numeric)
    mask = t.notna() & b.notna() & (t > DEFAULT_EPS) & (b > DEFAULT_EPS)
    ratio = t / b
    mask = mask & ratio.notna() & np.isfinite(ratio)
    return mask


def baseline_bin_table(
    target: np.ndarray,
    baseline: np.ndarray,
    generated: np.ndarray | None,
    n_bins: int,
    split_name: str,
) -> pd.DataFrame:
    edges = np.unique(np.quantile(baseline, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        return pd.DataFrame()
    bin_idx = np.digitize(baseline, edges[1:-1], right=False)
    rows: List[Dict[str, Any]] = []
    for b in range(int(bin_idx.max()) + 1):
        m = bin_idx == b
        if not np.any(m):
            continue
        y = target[m]
        pb = baseline[m]
        row: Dict[str, Any] = {
            "split": split_name,
            "bin": b,
            "rows": int(m.sum()),
            "baseline_lo": float(edges[b]) if b < len(edges) else float("nan"),
            "baseline_hi": float(edges[b + 1]) if b + 1 < len(edges) else float("nan"),
            "target_mean": float(np.mean(y)),
            "baseline_mean": float(np.mean(pb)),
            "baseline_mae": float(np.mean(np.abs(pb - y))),
        }
        if generated is not None:
            pg = generated[m]
            gm = compute_metrics(y, pg)
            bm = compute_metrics(y, pb)
            row.update(
                {
                    "generated_mae": gm["mae"],
                    "generated_rmse": gm["rmse"],
                    "generated_median_ae": gm["median_ae"],
                    "generated_bias": gm["bias"],
                    "mae_gain_vs_baseline": bm["mae"] - gm["mae"],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def top_feature_drift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    top_k: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for col in feature_cols:
        tr = train[col] if col in train.columns else pd.Series([np.nan] * len(train))
        te = test[col] if col in test.columns else pd.Series([np.nan] * len(test))
        tr_num = tr.map(extract_numeric)
        te_num = te.map(extract_numeric)
        tr_num_ratio = float(tr_num.notna().mean())
        if tr_num_ratio >= 0.8:
            drift = psi_from_train_edges(
                tr_num.to_numpy(dtype=float), te_num.to_numpy(dtype=float), bins=10
            )
            kind = "psi_numeric"
        else:
            drift = tv_categorical(tr, te)
            kind = "tv_categorical"
        rows.append({"feature": col, "drift_kind": kind, "drift_value": drift})
    out = pd.DataFrame(rows).sort_values("drift_value", ascending=False, na_position="last")
    return out.head(top_k).reset_index(drop=True)


def domain_classifier_summary(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, float]:
    if not HAS_SKLEARN:
        return {"available": 0.0}
    common = [c for c in feature_cols if c in train.columns and c in test.columns]
    if not common:
        return {"available": 0.0}
    x = pd.concat([train[common], test[common]], axis=0, ignore_index=True)
    y = np.concatenate([np.zeros(len(train), dtype=int), np.ones(len(test), dtype=int)])
    # Lightweight numeric encoding for mixed tables.
    x_enc = x.copy()
    for col in common:
        s = x_enc[col]
        if s.map(extract_numeric).notna().mean() >= 0.8:
            x_enc[col] = s.map(extract_numeric)
        else:
            x_enc[col] = s.astype(str).fillna("__nan__").astype("category").cat.codes
    x_enc = x_enc.fillna(-999)
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42,
    )
    clf.fit(x_enc, y)
    p_test = clf.predict_proba(x_enc.iloc[: len(train)])[:, 1]
    p_test = np.clip(p_test, 1e-4, 1.0 - 1e-4)
    w = p_test / (1.0 - p_test)
    w = np.clip(w, 1.0 / 8.0, 8.0)
    w = w / max(float(np.mean(w)), 1e-12)
    auc = float(roc_auc_score(y, clf.predict_proba(x_enc)[:, 1]))
    return {
        "available": 1.0,
        "domain_auc": auc,
        "weight_mean": float(np.mean(w)),
        "weight_p50": float(np.quantile(w, 0.5)),
        "weight_p90": float(np.quantile(w, 0.9)),
        "weight_p99": float(np.quantile(w, 0.99)),
        "weight_max": float(np.max(w)),
        "weight_gt2_frac": float(np.mean(w > 2.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Privacy-safe train/test gap diagnostics.")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE)
    parser.add_argument("--generated-file", default=DEFAULT_GENERATED_FILE)
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    parser.add_argument("--baseline-col", default=DEFAULT_BASELINE_COL)
    parser.add_argument("--generated-col", default=DEFAULT_GENERATED_COL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--drift-top-k", type=int, default=DEFAULT_DRIFT_TOP_K)
    parser.add_argument("--baseline-bins", type=int, default=DEFAULT_BASELINE_BINS)
    parser.add_argument("--skip-generated", action="store_true", help="Skip prediction error analysis.")
    args = parser.parse_args()

    train_path = Path(args.train_file)
    test_path = Path(args.test_file)
    gen_path = Path(args.generated_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        print(f"[ERROR] Train file not found: {train_path}")
        return 1
    if not test_path.exists():
        print(f"[ERROR] Test file not found: {test_path}")
        return 1

    train_df = load_table(train_path)
    test_df = load_table(test_path)
    train_df.columns = [str(c).strip() for c in train_df.columns]
    test_df.columns = [str(c).strip() for c in test_df.columns]

    for col in (args.target_col, args.baseline_col):
        if col not in train_df.columns:
            print(f"[ERROR] Train missing column: {col}")
            return 1

    feature_cols = [c for c in train_df.columns if c not in {args.target_col, args.baseline_col}]
    test_has_target = args.target_col in test_df.columns

    # ----- label vectors -----
    tr_target = train_df[args.target_col].map(extract_numeric)
    tr_base = train_df[args.baseline_col].map(extract_numeric)
    tr_valid = valid_label_mask(train_df[args.target_col], train_df[args.baseline_col])

    te_target = test_df[args.target_col].map(extract_numeric) if test_has_target else pd.Series([np.nan] * len(test_df))
    te_base = test_df[args.baseline_col].map(extract_numeric) if args.baseline_col in test_df.columns else pd.Series([np.nan] * len(test_df))
    te_valid = valid_label_mask(test_df[args.target_col], test_df[args.baseline_col]) if test_has_target else pd.Series([False] * len(test_df))

    tr_y = tr_target[tr_valid].to_numpy(dtype=float)
    tr_b = tr_base[tr_valid].to_numpy(dtype=float)
    tr_ratio = tr_y / tr_b
    tr_log_corr = signed_log1p(tr_y) - signed_log1p(tr_b)

    te_y = te_target[te_valid].to_numpy(dtype=float) if test_has_target else np.array([], dtype=float)
    te_b = te_base[te_valid].to_numpy(dtype=float) if test_has_target else np.array([], dtype=float)
    te_ratio = te_y / te_b if te_y.size else np.array([], dtype=float)

    summary: Dict[str, Any] = {
        "train_file": str(train_path),
        "test_file": str(test_path),
        "rows_total": {"train": int(len(train_df)), "test": int(len(test_df))},
        "rows_valid_positive": {"train": int(tr_valid.sum()), "test": int(te_valid.sum())},
        "feature_count_excl_target_baseline": int(len(feature_cols)),
        "column_overlap": {
            "train_only_cols": int(len(set(train_df.columns) - set(test_df.columns))),
            "test_only_cols": int(len(set(test_df.columns) - set(train_df.columns))),
            "shared_cols": int(len(set(train_df.columns) & set(test_df.columns))),
        },
        "label_quantiles": {
            "train_target": quantile_dict(tr_y, prefix=""),
            "test_target": quantile_dict(te_y, prefix=""),
            "train_baseline": quantile_dict(tr_b, prefix=""),
            "test_baseline": quantile_dict(te_b, prefix=""),
            "train_ratio_target_over_baseline": quantile_dict(tr_ratio, prefix=""),
            "test_ratio_target_over_baseline": quantile_dict(te_ratio, prefix=""),
            "train_log_correction": quantile_dict(tr_log_corr, prefix=""),
        },
        "label_shift_psi_train_edges": {
            "target": psi_from_train_edges(tr_y, te_y),
            "baseline": psi_from_train_edges(tr_b, te_b),
            "ratio": psi_from_train_edges(tr_ratio, te_ratio) if te_ratio.size else float("nan"),
        },
        "baseline_error_on_valid_rows": {
            "train_mae": float(np.mean(np.abs(tr_b - tr_y))) if tr_y.size else float("nan"),
            "test_mae": float(np.mean(np.abs(te_b - te_y))) if te_y.size else float("nan"),
            "train_median_ae": float(np.median(np.abs(tr_b - tr_y))) if tr_y.size else float("nan"),
            "test_median_ae": float(np.median(np.abs(te_b - te_y))) if te_y.size else float("nan"),
        },
        "domain_classifier": domain_classifier_summary(train_df, test_df, feature_cols),
    }

    # ----- drift report -----
    drift_df = top_feature_drift(train_df, test_df, feature_cols, top_k=args.drift_top_k)
    drift_path = out_dir / "top_feature_drift.csv"
    drift_df.to_csv(drift_path, index=False, encoding="utf-8-sig")
    summary["top_drift_features"] = drift_df.to_dict(orient="records")

    # ----- baseline bin population shift (train edges) -----
    edges = np.unique(np.quantile(tr_b, np.linspace(0.0, 1.0, args.baseline_bins + 1))) if tr_b.size else np.array([])
    pop_rows: List[Dict[str, Any]] = []
    if edges.size >= 2:
        tr_idx = np.digitize(tr_b, edges[1:-1], right=False)
        te_idx = np.digitize(te_b, edges[1:-1], right=False) if te_b.size else np.array([], dtype=int)
        for b in range(len(edges) - 1):
            tr_cnt = int(np.sum(tr_idx == b))
            te_cnt = int(np.sum(te_idx == b)) if te_idx.size else 0
            pop_rows.append(
                {
                    "bin": b,
                    "baseline_lo": float(edges[b]),
                    "baseline_hi": float(edges[b + 1]),
                    "train_rows": tr_cnt,
                    "test_rows": te_cnt,
                    "train_frac": float(tr_cnt / max(tr_idx.size, 1)),
                    "test_frac": float(te_cnt / max(te_idx.size, 1)) if te_idx.size else float("nan"),
                    "test_minus_train_frac": float(te_cnt / max(te_idx.size, 1) - tr_cnt / max(tr_idx.size, 1))
                    if te_idx.size
                    else float("nan"),
                }
            )
    pop_df = pd.DataFrame(pop_rows)
    pop_path = out_dir / "baseline_bin_population_shift.csv"
    pop_df.to_csv(pop_path, index=False, encoding="utf-8-sig")

    # ----- generated / error analysis -----
    tail_df = pd.DataFrame()
    seg_df = pd.DataFrame()
    if (not args.skip_generated) and gen_path.exists() and test_has_target:
        gen_df = load_table(gen_path)
        gen_df.columns = [str(c).strip() for c in gen_df.columns]
        if args.generated_col in gen_df.columns:
            g_target = gen_df[args.target_col].map(extract_numeric)
            g_base = gen_df[args.baseline_col].map(extract_numeric)
            g_pred = gen_df[args.generated_col].map(extract_numeric)
            g_valid = g_target.notna() & g_base.notna() & g_pred.notna()
            g_valid = g_valid & (g_target > DEFAULT_EPS) & (g_base > DEFAULT_EPS)
            y = g_target[g_valid].to_numpy(dtype=float)
            pb = g_base[g_valid].to_numpy(dtype=float)
            pg = g_pred[g_valid].to_numpy(dtype=float)

            bm = compute_metrics(y, pb)
            gm = compute_metrics(y, pg)
            summary["test_prediction_metrics"] = {
                "rows_used": int(len(y)),
                "baseline": bm,
                "generated": gm,
                "delta_generated_minus_baseline": {
                    k: gm[k] - bm[k] for k in ("mae", "rmse", "r2", "median_ae", "bias", "trimmed_mae90")
                },
            }
            summary["test_error_tail_baseline"] = error_tail_concentration(y, pb)
            summary["test_error_tail_generated"] = error_tail_concentration(y, pg)

            tail_rows: List[Dict[str, Any]] = []
            for row in summary["test_error_tail_generated"]:
                row = dict(row)
                row["model"] = "generated"
                tail_rows.append(row)
            for row in summary["test_error_tail_baseline"]:
                row = dict(row)
                row["model"] = "baseline"
                tail_rows.append(row)
            tail_df = pd.DataFrame(tail_rows)
            tail_path = out_dir / "error_tail_concentration.csv"
            tail_df.to_csv(tail_path, index=False, encoding="utf-8-sig")

            seg_df = baseline_bin_table(y, pb, pg, n_bins=args.baseline_bins, split_name="test")
            seg_path = out_dir / "test_metrics_by_baseline_bin.csv"
            seg_df.to_csv(seg_path, index=False, encoding="utf-8-sig")
        else:
            summary["test_prediction_metrics"] = {"error": f"missing column {args.generated_col}"}
    elif not gen_path.exists():
        summary["test_prediction_metrics"] = {"skipped": "generated file not found"}
    elif not test_has_target:
        summary["test_prediction_metrics"] = {"skipped": "test file has no target column"}

    summary_path = out_dir / "gap_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ----- human-readable stdout -----
    print("=== Privacy-Safe Train/Test Gap Diagnostics ===")
    print(f"Output dir: {out_dir.resolve()}")
    print(f"Train rows (valid): {summary['rows_valid_positive']['train']}")
    print(f"Test rows (valid):  {summary['rows_valid_positive']['test']}")
    print()
    print("--- Label shift (PSI, train-edges) ---")
    for k, v in summary["label_shift_psi_train_edges"].items():
        print(f"  {k}: {v:.4f}" if np.isfinite(v) else f"  {k}: NaN")
    print()
    print("--- Baseline error on valid rows ---")
    be = summary["baseline_error_on_valid_rows"]
    print(f"  train MAE / median: {be['train_mae']:.2f} / {be['train_median_ae']:.2f}")
    print(f"  test  MAE / median: {be['test_mae']:.2f} / {be['test_median_ae']:.2f}")
    print()
    dom = summary["domain_classifier"]
    if dom.get("available", 0) == 1.0:
        print("--- Domain classifier (train vs test features) ---")
        print(f"  AUC: {dom['domain_auc']:.4f}")
        print(
            f"  importance weights: mean={dom['weight_mean']:.3f}, "
            f"p90={dom['weight_p90']:.3f}, max={dom['weight_max']:.3f}, "
            f"frac>2={dom['weight_gt2_frac']:.3f}"
        )
        print()
    if "generated" in summary.get("test_prediction_metrics", {}):
        tm = summary["test_prediction_metrics"]
        print("--- Test prediction metrics ---")
        print(f"  rows: {tm['rows_used']}")
        print(f"  baseline  MAE={tm['baseline']['mae']:.2f}, median={tm['baseline']['median_ae']:.2f}")
        print(f"  generated MAE={tm['generated']['mae']:.2f}, median={tm['generated']['median_ae']:.2f}")
        d = tm["delta_generated_minus_baseline"]
        print(
            f"  delta(MAE)={d['mae']:.2f}, delta(median)={d['median_ae']:.2f}, "
            f"delta(trimmed_mae90)={d['trimmed_mae90']:.2f}"
        )
        print()
        print("--- Error tail concentration (generated) ---")
        for row in summary["test_error_tail_generated"]:
            print(
                f"  top {int(row['top_fraction'] * 100):2d}% rows -> "
                f"{row['mae_share'] * 100:5.1f}% of total MAE, "
                f"mean |err|={row['mean_abs_err_in_top']:.2f}"
            )
    print()
    print("Files written:")
    print(f"  {summary_path.name}")
    print(f"  {drift_path.name}")
    print(f"  {pop_path.name}")
    if not tail_df.empty:
        print("  error_tail_concentration.csv")
    if not seg_df.empty:
        print("  test_metrics_by_baseline_bin.csv")
    print()
    print("Please share gap_summary.json and the CSV files (no raw data inside).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
