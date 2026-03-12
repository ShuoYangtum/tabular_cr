#!/usr/bin/env python3
"""
Visualize why generated outperforms baseline on target.

Outputs:
- overview figure (normalized metrics + distribution views)
- PCA projection figure
- 2D improvement heatmap figure
- metrics CSV and binned-stats CSV
"""

from __future__ import annotations

import argparse
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
DEFAULT_OUT_FIG = "prediction_advantage_overview.png"
DEFAULT_OUT_CSV = "prediction_advantage_metrics.csv"
DEFAULT_OUT_DIR = "prediction_advantage_plots"


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


def ecdf(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.sort(v)
    p = np.arange(1, len(s) + 1) / len(s)
    return s, p


def pca_2d(x: np.ndarray) -> np.ndarray:
    x0 = x - np.mean(x, axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x0, full_matrices=False)
    return u[:, :2] * s[:2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize generated vs baseline advantage.")
    parser.add_argument("--file", default=DEFAULT_FILE, help="Input file (csv/xlsx/xls)")
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    parser.add_argument("--generated-col", default=DEFAULT_GENERATED_COL)
    parser.add_argument("--baseline-col", default=DEFAULT_BASELINE_COL)
    parser.add_argument("--out-fig", default=DEFAULT_OUT_FIG)
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--heatmap-bins", type=int, default=10)
    parser.add_argument("--pca-max-points", type=int, default=12000)
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(args.out_csv, index=False)

    # ---- Figure 1: Overview ----
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Panel 1: normalized lower-is-better metrics (relative to baseline)
    show_keys = ["mae", "rmse", "median_ae", "p90_ae", "trimmed_rmse90"]
    x = np.arange(len(show_keys), dtype=float)
    gen_ratio = [m_gen[k] / max(abs(m_base[k]), 1e-12) for k in show_keys]
    base_ratio = [1.0 for _ in show_keys]
    axes[0, 0].bar(x - 0.18, gen_ratio, width=0.36, label="generated / baseline")
    axes[0, 0].bar(x + 0.18, base_ratio, width=0.36, label="baseline (1.0)")
    axes[0, 0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(show_keys, rotation=20)
    axes[0, 0].set_ylabel("normalized value (lower is better)")
    axes[0, 0].set_title("Normalized Lower-is-better Metrics")
    axes[0, 0].legend()

    # Panel 2: residual histogram
    lo = np.nanquantile(np.concatenate([err_gen, err_base]), 0.01)
    hi = np.nanquantile(np.concatenate([err_gen, err_base]), 0.99)
    bins = np.linspace(lo, hi, 80)
    axes[0, 1].hist(err_base, bins=bins, alpha=0.45, label="baseline residual")
    axes[0, 1].hist(err_gen, bins=bins, alpha=0.45, label="generated residual")
    axes[0, 1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Residual Distribution (1%-99% range)")
    axes[0, 1].set_xlabel("residual = pred - target")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].legend()

    # Panel 3: ECDF of absolute error
    xg, pg = ecdf(ae_gen)
    xb, pb = ecdf(ae_base)
    axes[1, 0].plot(xg, pg, label="generated |error|")
    axes[1, 0].plot(xb, pb, label="baseline |error|")
    axes[1, 0].set_title("Absolute Error ECDF (left is better)")
    axes[1, 0].set_xlabel("|pred-target|")
    axes[1, 0].set_ylabel("cdf")
    axes[1, 0].legend()

    # Panel 4: win-rate bar
    gen_better = float(np.mean(ae_gen < ae_base))
    tie_rate = float(np.mean(ae_gen == ae_base))
    base_better = float(np.mean(ae_gen > ae_base))
    axes[1, 1].bar(["generated better", "tie", "baseline better"], [gen_better, tie_rate, base_better])
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("ratio")
    axes[1, 1].set_title("Sample-level Winner Ratio")

    fig.suptitle(
        f"Generated vs Baseline (n={len(cleaned)}, "
        f"MAE gain={(m_base['mae'] - m_gen['mae']) / max(abs(m_base['mae']), 1e-12):.2%}, "
        f"RMSE gain={(m_base['rmse'] - m_gen['rmse']) / max(abs(m_base['rmse']), 1e-12):.2%})"
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    overview_path = out_dir / args.out_fig
    fig.savefig(overview_path, dpi=170)
    plt.close(fig)

    # ---- Figure 2: PCA projection ----
    n = len(cleaned)
    max_points = min(int(args.pca_max_points), n)
    if max_points < n:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n, size=max_points, replace=False))
    else:
        idx = np.arange(n)
    x_gen = np.column_stack([y[idx], p_gen[idx], np.abs(err_gen[idx]), err_gen[idx]])
    x_base = np.column_stack([y[idx], p_base[idx], np.abs(err_base[idx]), err_base[idx]])
    x_all = np.vstack([x_gen, x_base])
    z_all = pca_2d(x_all)
    z_gen = z_all[: len(idx)]
    z_base = z_all[len(idx) :]

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    ax2.scatter(z_base[:, 0], z_base[:, 1], s=8, alpha=0.20, label="baseline points")
    ax2.scatter(z_gen[:, 0], z_gen[:, 1], s=8, alpha=0.20, label="generated points")
    ax2.set_title("PCA Projection of [target, pred, |err|, err]")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.legend()
    pca_path = out_dir / "pca_projection.png"
    fig2.tight_layout()
    fig2.savefig(pca_path, dpi=170)
    plt.close(fig2)

    # ---- Figure 3: 2D heatmap of MAE gain ----
    bins = max(int(args.heatmap_bins), 3)
    y_q = pd.qcut(pd.Series(y), q=bins, duplicates="drop")
    b_q = pd.qcut(pd.Series(p_base), q=bins, duplicates="drop")
    hdf = pd.DataFrame({"y_bin": y_q.astype("string"), "b_bin": b_q.astype("string"), "ae_g": ae_gen, "ae_b": ae_base})
    agg = (
        hdf.groupby(["y_bin", "b_bin"], observed=False)
        .agg(mae_gen=("ae_g", "mean"), mae_base=("ae_b", "mean"), n=("ae_g", "size"))
        .reset_index()
    )
    agg["gain"] = agg["mae_base"] - agg["mae_gen"]  # positive means generated better
    pivot_gain = agg.pivot(index="y_bin", columns="b_bin", values="gain")
    pivot_n = agg.pivot(index="y_bin", columns="b_bin", values="n")
    heat = pivot_gain.to_numpy(dtype=float)
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 7))
    im = ax3.imshow(heat, aspect="auto", cmap="RdYlGn")
    ax3.set_title("Heatmap: MAE Gain (baseline - generated) by target/bin")
    ax3.set_xlabel("baseline quantile bins")
    ax3.set_ylabel("target quantile bins")
    ax3.set_xticks(np.arange(pivot_gain.shape[1]))
    ax3.set_yticks(np.arange(pivot_gain.shape[0]))
    ax3.set_xticklabels([str(c) for c in pivot_gain.columns], rotation=45, ha="right", fontsize=7)
    ax3.set_yticklabels([str(r) for r in pivot_gain.index], fontsize=7)
    cbar = fig3.colorbar(im, ax=ax3)
    cbar.set_label("MAE gain (positive is better)")
    # annotate sample count for readability
    nvals = pivot_n.to_numpy(dtype=float)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            if np.isfinite(nvals[i, j]) and nvals[i, j] > 0:
                ax3.text(j, i, int(nvals[i, j]), ha="center", va="center", fontsize=6, color="black")
    fig3.tight_layout()
    heatmap_path = out_dir / "improvement_heatmap.png"
    fig3.savefig(heatmap_path, dpi=170)
    plt.close(fig3)

    # Save binned stats for auditability
    agg.to_csv(out_dir / "improvement_heatmap_stats.csv", index=False)

    print(f"Saved overview figure: {overview_path}")
    print(f"Saved PCA figure: {pca_path}")
    print(f"Saved heatmap figure: {heatmap_path}")
    print(f"Saved metrics CSV: {args.out_csv}")
    print(f"Saved binned stats CSV: {out_dir / 'improvement_heatmap_stats.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

