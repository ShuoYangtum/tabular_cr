#!/usr/bin/env python3
"""
Professional visualization pack for generated vs baseline against target.

This script creates report-ready materials:
- executive dashboard
- parity/fit chart (target line y=x)
- diverging improvement heatmaps (positive and negative clearly visible)
- quantile gain chart
- supporting CSV tables
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


def ecdf(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.sort(v)
    p = np.arange(1, len(s) + 1) / len(s)
    return s, p


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    ae = np.abs(err)
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - np.sum((y_true - y_pred) ** 2) / ss_tot) if len(y_true) >= 2 and ss_tot > 0 else np.nan
    lo = int(0.05 * len(err))
    hi = int(0.95 * len(err))
    sorted_sq = np.sort(err**2)
    trimmed_rmse90 = float(np.sqrt(np.mean(sorted_sq[lo:hi]))) if hi > lo else np.nan
    return {
        "count_used": float(len(y_true)),
        "mae": float(np.mean(ae)),
        "rmse": rmse,
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.quantile(ae, 0.90)),
        "trimmed_rmse90": trimmed_rmse90,
        "bias": float(np.mean(err)),
        "r2": r2,
    }


def robust_axis_limits(*arrays: np.ndarray, q_low: float = 0.01, q_high: float = 0.99) -> tuple[float, float]:
    merged = np.concatenate([np.asarray(a, dtype=float) for a in arrays if len(a) > 0])
    lo = float(np.nanquantile(merged, q_low))
    hi = float(np.nanquantile(merged, q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(np.nanmin(merged))
        hi = float(np.nanmax(merged))
    if lo == hi:
        hi = lo + 1.0
    return lo, hi


def main() -> int:
    parser = argparse.ArgumentParser(description="Create report-grade generated vs baseline visualizations.")
    parser.add_argument("--file", default=DEFAULT_FILE, help="Input file (.csv/.xlsx/.xls)")
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    parser.add_argument("--generated-col", default=DEFAULT_GENERATED_COL)
    parser.add_argument("--baseline-col", default=DEFAULT_BASELINE_COL)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--bins", type=int, default=10, help="Quantile bins for heatmaps and decile analyses.")
    parser.add_argument("--scatter-max-points", type=int, default=15000)
    parser.add_argument("--topk-waterfall", type=int, default=30, help="Top-K hardest samples for waterfall chart.")
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
    delta_ae = ae_base - ae_gen  # positive means generated better

    m_gen = compute_metrics(y, p_gen)
    m_base = compute_metrics(y, p_base)

    # -------- metrics table --------
    keys = ["count_used", "mae", "rmse", "median_ae", "p90_ae", "trimmed_rmse90", "bias", "r2"]
    rows = []
    for k in keys:
        g = m_gen[k]
        b = m_base[k]
        rel = (b - g) / max(abs(b), 1e-12) if k != "r2" else (g - b) / max(abs(b), 1e-12)
        rows.append({"metric": k, "generated": g, "baseline": b, "relative_gain": rel})
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "metrics_comparison.csv", index=False)

    # -------- Figure 1: Executive dashboard --------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # A1 normalized metrics
    show_keys = ["mae", "rmse", "median_ae", "p90_ae", "trimmed_rmse90"]
    x = np.arange(len(show_keys))
    gen_norm = [m_gen[k] / max(abs(m_base[k]), 1e-12) for k in show_keys]
    base_norm = [1.0 for _ in show_keys]
    axes[0, 0].bar(x - 0.18, gen_norm, width=0.36, label="generated / baseline")
    axes[0, 0].bar(x + 0.18, base_norm, width=0.36, label="baseline = 1")
    axes[0, 0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(show_keys, rotation=20)
    axes[0, 0].set_title("Normalized Lower-is-better Metrics")
    axes[0, 0].set_ylabel("ratio")
    axes[0, 0].legend()

    # A2 residual distribution
    lo, hi = robust_axis_limits(err_gen, err_base)
    bins = np.linspace(lo, hi, 80)
    axes[0, 1].hist(err_base, bins=bins, alpha=0.45, label="baseline residual")
    axes[0, 1].hist(err_gen, bins=bins, alpha=0.45, label="generated residual")
    axes[0, 1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Residual Distribution")
    axes[0, 1].set_xlabel("pred - target")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].legend()

    # A3 ECDF abs error
    xg, pg = ecdf(ae_gen)
    xb, pb = ecdf(ae_base)
    axes[0, 2].plot(xg, pg, label="generated")
    axes[0, 2].plot(xb, pb, label="baseline")
    axes[0, 2].set_title("Absolute Error ECDF (left is better)")
    axes[0, 2].set_xlabel("|pred-target|")
    axes[0, 2].set_ylabel("cdf")
    axes[0, 2].legend()

    # A4 win rate
    gen_better = float(np.mean(ae_gen < ae_base))
    tie_rate = float(np.mean(ae_gen == ae_base))
    base_better = float(np.mean(ae_gen > ae_base))
    axes[1, 0].bar(["generated better", "tie", "baseline better"], [gen_better, tie_rate, base_better])
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_ylabel("ratio")
    axes[1, 0].set_title("Sample-level Winner Ratio")

    # A5 calibration/binned fit
    q = max(int(args.bins), 4)
    yb = pd.qcut(pd.Series(y), q=q, duplicates="drop")
    bdf = pd.DataFrame({"y": y, "g": p_gen, "b": p_base, "yb": yb})
    agg_fit = bdf.groupby("yb", observed=False).agg(y=("y", "mean"), g=("g", "mean"), b=("b", "mean"), n=("y", "size"))
    axes[1, 1].plot(agg_fit["y"], agg_fit["y"], "k--", label="ideal y=x")
    axes[1, 1].plot(agg_fit["y"], agg_fit["b"], marker="o", label="baseline bin-mean")
    axes[1, 1].plot(agg_fit["y"], agg_fit["g"], marker="o", label="generated bin-mean")
    axes[1, 1].set_title("Binned Calibration (target vs prediction)")
    axes[1, 1].set_xlabel("target mean per bin")
    axes[1, 1].set_ylabel("prediction mean per bin")
    axes[1, 1].legend()

    # A6 quantile gain curve
    dec = pd.qcut(pd.Series(y), q=q, duplicates="drop")
    gdf = pd.DataFrame({"decile": dec.astype("string"), "ae_g": ae_gen, "ae_b": ae_base})
    agg_gain = gdf.groupby("decile", observed=False).agg(mae_gen=("ae_g", "mean"), mae_base=("ae_b", "mean"), n=("ae_g", "size"))
    agg_gain["gain"] = agg_gain["mae_base"] - agg_gain["mae_gen"]
    axes[1, 2].bar(np.arange(len(agg_gain)), agg_gain["gain"].to_numpy())
    axes[1, 2].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1, 2].set_xticks(np.arange(len(agg_gain)))
    axes[1, 2].set_xticklabels(agg_gain.index.astype(str), rotation=35, ha="right", fontsize=8)
    axes[1, 2].set_title("MAE Gain by Target Quantile (baseline - generated)")
    axes[1, 2].set_ylabel("gain (>0 is better)")

    fig.suptitle(
        f"Executive Dashboard (n={len(cleaned)} | "
        f"MAE gain={(m_base['mae']-m_gen['mae'])/max(abs(m_base['mae']),1e-12):.2%}, "
        f"RMSE gain={(m_base['rmse']-m_gen['rmse'])/max(abs(m_base['rmse']),1e-12):.2%})"
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out_dir / "01_executive_dashboard.png", dpi=170)
    plt.close(fig)

    # -------- Figure 2: Target-line fit (professional parity chart) --------
    n = len(y)
    m = min(n, int(args.scatter_max_points))
    if m < n:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n, size=m, replace=False))
    else:
        idx = np.arange(n)
    ys = y[idx]
    gs = p_gen[idx]
    bs = p_base[idx]
    lo2, hi2 = robust_axis_limits(ys, gs, bs)

    fig2, ax = plt.subplots(1, 1, figsize=(9, 7))
    hb1 = ax.hexbin(ys, bs, gridsize=45, bins="log", alpha=0.45, cmap="Blues", mincnt=1, label="baseline density")
    hb2 = ax.hexbin(ys, gs, gridsize=45, bins="log", alpha=0.45, cmap="Reds", mincnt=1, label="generated density")
    ax.plot([lo2, hi2], [lo2, hi2], "k--", linewidth=1.5, label="ideal y=x")

    # trend lines
    k_b, b_b = np.polyfit(ys, bs, deg=1)
    k_g, b_g = np.polyfit(ys, gs, deg=1)
    xx = np.linspace(lo2, hi2, 200)
    ax.plot(xx, k_b * xx + b_b, color="navy", linewidth=2, label=f"baseline fit: y={k_b:.3f}x+{b_b:.3f}")
    ax.plot(xx, k_g * xx + b_g, color="darkred", linewidth=2, label=f"generated fit: y={k_g:.3f}x+{b_g:.3f}")

    ax.set_xlim(lo2, hi2)
    ax.set_ylim(lo2, hi2)
    ax.set_xlabel("target")
    ax.set_ylabel("prediction")
    ax.set_title("Target-Line Fit View (closer to y=x is better)")
    ax.legend(loc="upper left", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(out_dir / "02_target_line_fit.png", dpi=170)
    plt.close(fig2)

    # -------- Figure 3: Diverging heatmaps (with positive and negative contrast) --------
    bins = max(int(args.bins), 4)
    y_q = pd.qcut(pd.Series(y), q=bins, duplicates="drop")
    b_q = pd.qcut(pd.Series(p_base), q=bins, duplicates="drop")
    hdf = pd.DataFrame(
        {
            "y_bin": y_q.astype("string"),
            "b_bin": b_q.astype("string"),
            "err_g": err_gen,
            "err_b": err_base,
            "ae_g": ae_gen,
            "ae_b": ae_base,
            "delta_ae": delta_ae,
        }
    )
    agg = (
        hdf.groupby(["y_bin", "b_bin"], observed=False)
        .agg(
            mean_gain=("delta_ae", "mean"),
            median_gain=("delta_ae", "median"),
            mean_err_gen=("err_g", "mean"),
            mean_err_base=("err_b", "mean"),
            n=("delta_ae", "size"),
        )
        .reset_index()
    )
    agg["bias_shift"] = agg["mean_err_base"] - agg["mean_err_gen"]

    heat_mean = agg.pivot(index="y_bin", columns="b_bin", values="mean_gain")
    heat_bias = agg.pivot(index="y_bin", columns="b_bin", values="bias_shift")
    heat_n = agg.pivot(index="y_bin", columns="b_bin", values="n")

    fig3, axarr = plt.subplots(1, 2, figsize=(15, 6))
    for ax, mat_df, title in [
        (axarr[0], heat_mean, "Mean |error| Gain (baseline - generated)"),
        (axarr[1], heat_bias, "Bias-Shift Gain (err_base - err_gen)"),
    ]:
        mat = mat_df.to_numpy(dtype=float)
        vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
        if vmax <= 0 or not np.isfinite(vmax):
            vmax = 1.0
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("baseline quantile bins")
        ax.set_ylabel("target quantile bins")
        ax.set_xticks(np.arange(mat_df.shape[1]))
        ax.set_yticks(np.arange(mat_df.shape[0]))
        ax.set_xticklabels([str(c) for c in mat_df.columns], rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels([str(r) for r in mat_df.index], fontsize=7)
        nvals = heat_n.to_numpy(dtype=float)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(nvals[i, j]) and nvals[i, j] > 0:
                    ax.text(j, i, int(nvals[i, j]), ha="center", va="center", fontsize=6, color="black")
        cbar = fig3.colorbar(im, ax=ax)
        cbar.set_label("gain (red/blue around 0)")
    fig3.tight_layout()
    fig3.savefig(out_dir / "03_diverging_heatmaps.png", dpi=170)
    plt.close(fig3)

    # -------- Figure 4: Quantile gain line --------
    qdf = pd.DataFrame({"target": y, "delta_ae": delta_ae, "ae_g": ae_gen, "ae_b": ae_base})
    qdf["q"] = pd.qcut(qdf["target"], q=bins, duplicates="drop").astype("string")
    qagg = (
        qdf.groupby("q", observed=False)
        .agg(mean_gain=("delta_ae", "mean"), median_gain=("delta_ae", "median"), mae_gen=("ae_g", "mean"), mae_base=("ae_b", "mean"), n=("delta_ae", "size"))
        .reset_index()
    )
    fig4, ax4 = plt.subplots(1, 1, figsize=(12, 5))
    ax4.plot(qagg["q"], qagg["mean_gain"], marker="o", label="mean gain")
    ax4.plot(qagg["q"], qagg["median_gain"], marker="o", label="median gain")
    ax4.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax4.set_title("Gain by Target Quantile")
    ax4.set_xlabel("target quantile bin")
    ax4.set_ylabel("gain = |err_base| - |err_gen|")
    ax4.tick_params(axis="x", rotation=35)
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(out_dir / "04_quantile_gain_curve.png", dpi=170)
    plt.close(fig4)

    # -------- Figure 5: Cumulative gain / lift curves --------
    cdf = pd.DataFrame(
        {
            "target": y,
            "ae_base": ae_base,
            "ae_gen": ae_gen,
            "delta_ae": delta_ae,  # baseline - generated
        }
    )
    # Focus on business-critical points first: hardest baseline cases.
    cdf_hard = cdf.sort_values("ae_base", ascending=False).reset_index(drop=True)
    cdf_hard["cum_delta"] = cdf_hard["delta_ae"].cumsum()
    cdf_hard["k"] = np.arange(1, len(cdf_hard) + 1)
    cdf_hard["cum_avg_delta"] = cdf_hard["cum_delta"] / cdf_hard["k"]

    # Alternative view: cumulative gain along target scale.
    cdf_target = cdf.sort_values("target", ascending=True).reset_index(drop=True)
    cdf_target["cum_delta"] = cdf_target["delta_ae"].cumsum()
    cdf_target["k"] = np.arange(1, len(cdf_target) + 1)
    cdf_target["cum_avg_delta"] = cdf_target["cum_delta"] / cdf_target["k"]

    fig5, ax5 = plt.subplots(1, 2, figsize=(15, 5))
    ax5[0].plot(cdf_hard["k"], cdf_hard["cum_delta"], label="sorted by baseline |error| desc", linewidth=2)
    ax5[0].plot(cdf_target["k"], cdf_target["cum_delta"], label="sorted by target asc", linewidth=2)
    ax5[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax5[0].set_title("Cumulative Gain Curve")
    ax5[0].set_xlabel("Top-k samples included")
    ax5[0].set_ylabel("Cumulative gain: sum(|err_base|-|err_gen|)")
    ax5[0].legend()

    ax5[1].plot(cdf_hard["k"], cdf_hard["cum_avg_delta"], label="hard-sample lift", linewidth=2)
    ax5[1].plot(cdf_target["k"], cdf_target["cum_avg_delta"], label="target-order lift", linewidth=2)
    ax5[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax5[1].set_title("Average Gain Lift Curve")
    ax5[1].set_xlabel("Top-k samples included")
    ax5[1].set_ylabel("Average gain per sample")
    ax5[1].legend()
    fig5.tight_layout()
    fig5.savefig(out_dir / "05_cumulative_gain_lift.png", dpi=170)
    plt.close(fig5)

    # -------- Figure 6: Top-K hardest samples waterfall --------
    topk = max(5, int(args.topk_waterfall))
    hardest = (
        pd.DataFrame(
            {
                "target": y,
                "baseline": p_base,
                "generated": p_gen,
                "ae_base": ae_base,
                "ae_gen": ae_gen,
                "delta_ae": delta_ae,
            }
        )
        .sort_values("ae_base", ascending=False)
        .head(topk)
        .reset_index(drop=True)
    )
    hardest["rank"] = np.arange(1, len(hardest) + 1)

    fig6, ax6 = plt.subplots(1, 1, figsize=(13, 5))
    colors = ["tab:green" if v >= 0 else "tab:red" for v in hardest["delta_ae"].to_numpy()]
    ax6.bar(hardest["rank"], hardest["delta_ae"], color=colors, alpha=0.85)
    ax6.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax6.set_title("Top-K Hardest Baseline Samples: Gain Waterfall")
    ax6.set_xlabel("Hard-sample rank (1 = hardest by baseline |error|)")
    ax6.set_ylabel("gain = |err_base| - |err_gen|")
    fig6.tight_layout()
    fig6.savefig(out_dir / "06_topk_hardcase_waterfall.png", dpi=170)
    plt.close(fig6)

    # -------- Figure 7: Hard-sample parity zoom --------
    hz = hardest.copy()
    loz, hiz = robust_axis_limits(hz["target"].to_numpy(), hz["generated"].to_numpy(), hz["baseline"].to_numpy(), q_low=0.0, q_high=1.0)
    fig7, ax7 = plt.subplots(1, 1, figsize=(8, 6))
    ax7.scatter(hz["target"], hz["baseline"], s=40, alpha=0.8, label="baseline (hard cases)")
    ax7.scatter(hz["target"], hz["generated"], s=40, alpha=0.8, label="generated (hard cases)")
    ax7.plot([loz, hiz], [loz, hiz], "k--", linewidth=1.5, label="ideal y=x")
    ax7.set_xlim(loz, hiz)
    ax7.set_ylim(loz, hiz)
    ax7.set_title("Hard-case Parity Zoom")
    ax7.set_xlabel("target")
    ax7.set_ylabel("prediction")
    ax7.legend()
    fig7.tight_layout()
    fig7.savefig(out_dir / "07_hardcase_parity_zoom.png", dpi=170)
    plt.close(fig7)

    # Save supporting tables
    agg.to_csv(out_dir / "heatmap_stats.csv", index=False)
    qagg.to_csv(out_dir / "quantile_gain_stats.csv", index=False)
    cdf_hard.to_csv(out_dir / "cumulative_gain_hard_sorted.csv", index=False)
    cdf_target.to_csv(out_dir / "cumulative_gain_target_sorted.csv", index=False)
    hardest.to_csv(out_dir / "topk_hardcase_gain.csv", index=False)

    print(f"Saved report figures to: {out_dir}")
    print("- 01_executive_dashboard.png")
    print("- 02_target_line_fit.png")
    print("- 03_diverging_heatmaps.png")
    print("- 04_quantile_gain_curve.png")
    print("- 05_cumulative_gain_lift.png")
    print("- 06_topk_hardcase_waterfall.png")
    print("- 07_hardcase_parity_zoom.png")
    print("Saved tables:")
    print("- metrics_comparison.csv")
    print("- heatmap_stats.csv")
    print("- quantile_gain_stats.csv")
    print("- cumulative_gain_hard_sorted.csv")
    print("- cumulative_gain_target_sorted.csv")
    print("- topk_hardcase_gain.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

