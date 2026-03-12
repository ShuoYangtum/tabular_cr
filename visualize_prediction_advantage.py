#!/usr/bin/env python3
"""
Visualize why generated outperforms baseline on target.

Outputs:
- A comparison figure (metrics, residual distribution, abs-error ECDF)
- A CSV with metric values and relative improvements
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

DEFAULT_FILE = "generated.csv"
DEFAULT_TARGET_COL = "esg_firma_esg-bewertung__input__wasserverbrauch-m3"
DEFAULT_GENERATED_COL = "generated"
DEFAULT_BASELINE_COL = "esg_firma_wasser_berechnet"
DEFAULT_OUT_FIG = "prediction_advantage.png"
DEFAULT_OUT_CSV = "prediction_advantage_metrics.csv"


def extract_numeric(value) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        v = float(value)
        return v if np.isfinite(v) else np.nan
    s = str(value).strip()
    if not s:
        return np.nan
    s = s.replace(",", "")
    s = re.sub(r"\s+", "", s)
    if s.lower() in {"nan", "none", "null", "na", "n/a", "-", "--"}:
        return np.nan
    m = NUMERIC_PATTERN.search(s)
    if not m:
        return np.nan
    try:
        v = float(m.group())
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    ae = np.abs(err)
    rmse = float(np.sqrt(np.mean(err**2)))
    return {
        "count_used": float(len(y_true)),
        "mae": float(np.mean(ae)),
        "rmse": rmse,
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.quantile(ae, 0.90)),
        "trimmed_rmse90": float(np.sqrt(np.mean(np.sort(err**2)[int(0.05 * len(err)): int(0.95 * len(err))]))),
        "bias": float(np.mean(err)),
        "r2": float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        if len(y_true) >= 2 and np.sum((y_true - np.mean(y_true)) ** 2) > 0
        else np.nan,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize generated vs baseline advantage.")
    parser.add_argument("--file", default=DEFAULT_FILE, help="Input file (csv/xlsx/xls)")
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    parser.add_argument("--generated-col", default=DEFAULT_GENERATED_COL)
    parser.add_argument("--baseline-col", default=DEFAULT_BASELINE_COL)
    parser.add_argument("--out-fig", default=DEFAULT_OUT_FIG)
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path, low_memory=False)
    elif file_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Only csv/xlsx/xls are supported.")

    for c in [args.target_col, args.generated_col, args.baseline_col]:
        if c not in df.columns:
            raise KeyError(f"Column not found: {c}")

    cleaned = pd.DataFrame(
        {
            "target": df[args.target_col].map(extract_numeric),
            "generated": df[args.generated_col].map(extract_numeric),
            "baseline": df[args.baseline_col].map(extract_numeric),
        }
    ).dropna(subset=["target", "generated", "baseline"])

    if len(cleaned) == 0:
        raise ValueError("No valid rows after numeric cleaning.")

    y = cleaned["target"].to_numpy(dtype=float)
    p_gen = cleaned["generated"].to_numpy(dtype=float)
    p_base = cleaned["baseline"].to_numpy(dtype=float)
    err_gen = p_gen - y
    err_base = p_base - y
    ae_gen = np.abs(err_gen)
    ae_base = np.abs(err_base)

    m_gen = compute_metrics(y, p_gen)
    m_base = compute_metrics(y, p_base)

    metric_rows = []
    keys = ["count_used", "mae", "rmse", "median_ae", "p90_ae", "trimmed_rmse90", "bias", "r2"]
    for k in keys:
        g = m_gen[k]
        b = m_base[k]
        if k in {"r2"}:
            rel = (g - b) / max(abs(b), 1e-12) if np.isfinite(b) else np.nan
        else:
            rel = (b - g) / max(abs(b), 1e-12) if np.isfinite(b) else np.nan
        metric_rows.append({"metric": k, "generated": g, "baseline": b, "relative_gain": rel})
    pd.DataFrame(metric_rows).to_csv(args.out_csv, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: key metrics
    show_keys = ["mae", "rmse", "median_ae", "p90_ae", "trimmed_rmse90"]
    x = np.arange(len(show_keys))
    axes[0].bar(x - 0.18, [m_gen[k] for k in show_keys], width=0.36, label="generated")
    axes[0].bar(x + 0.18, [m_base[k] for k in show_keys], width=0.36, label="baseline")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(show_keys, rotation=20)
    axes[0].set_title("Lower-is-better metrics")
    axes[0].legend()

    # Panel 2: residual histogram
    lo = np.nanquantile(np.concatenate([err_gen, err_base]), 0.01)
    hi = np.nanquantile(np.concatenate([err_gen, err_base]), 0.99)
    bins = np.linspace(lo, hi, 80)
    axes[1].hist(err_base, bins=bins, alpha=0.45, label="baseline residual")
    axes[1].hist(err_gen, bins=bins, alpha=0.45, label="generated residual")
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Residual distribution (1%-99% range)")
    axes[1].set_xlabel("residual = pred - target")
    axes[1].set_ylabel("count")
    axes[1].legend()

    # Panel 3: ECDF of absolute error
    def ecdf(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.sort(v)
        p = np.arange(1, len(s) + 1) / len(s)
        return s, p

    xg, pg = ecdf(ae_gen)
    xb, pb = ecdf(ae_base)
    axes[2].plot(xg, pg, label="generated |error|")
    axes[2].plot(xb, pb, label="baseline |error|")
    axes[2].set_title("Absolute error ECDF (left is better)")
    axes[2].set_xlabel("|pred-target|")
    axes[2].set_ylabel("cdf")
    axes[2].legend()

    fig.suptitle(
        f"Generated vs Baseline (n={len(cleaned)}, "
        f"MAE gain={(m_base['mae'] - m_gen['mae']) / max(abs(m_base['mae']), 1e-12):.2%}, "
        f"RMSE gain={(m_base['rmse'] - m_gen['rmse']) / max(abs(m_base['rmse']), 1e-12):.2%})"
    )
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=160)
    print(f"Saved figure: {args.out_fig}")
    print(f"Saved metrics: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

