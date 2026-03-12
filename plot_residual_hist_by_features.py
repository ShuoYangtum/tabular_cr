#!/usr/bin/env python3
"""
Plot residual histograms by feature bins/categories.

For each selected feature:
- x-axis: residual (pred - target)
- y-axis: count
- Facets: bins (numeric) or categories (categorical)
- Compare generated vs baseline in each facet
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

DEFAULT_FILE = "generated.csv"
DEFAULT_FEATURES_FILE = "bug.txt"
DEFAULT_TARGET_COL = "esg_firma_esg-bewertung__input__wasserverbrauch-m3"
DEFAULT_GENERATED_COL = "generated"
DEFAULT_BASELINE_COL = "esg_firma_wasser_berechnet"
DEFAULT_OUT_DIR = "residual_hist_by_feature"


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


def sanitize_filename(s: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return out[:180]


def read_features(features_file: Path) -> List[str]:
    lines = [ln.strip() for ln in features_file.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln]


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot residual histograms conditioned on selected features.")
    parser.add_argument("--file", default=DEFAULT_FILE)
    parser.add_argument("--features-file", default=DEFAULT_FEATURES_FILE)
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    parser.add_argument("--generated-col", default=DEFAULT_GENERATED_COL)
    parser.add_argument("--baseline-col", default=DEFAULT_BASELINE_COL)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-bins", type=int, default=6, help="Max numeric bins / top categories per feature.")
    parser.add_argument("--numeric-like-threshold", type=float, default=0.75)
    args = parser.parse_args()

    file_path = Path(args.file)
    features_path = Path(args.features_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path, low_memory=False)
    elif file_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Only csv/xlsx/xls are supported.")

    for c in [args.target_col, args.generated_col, args.baseline_col]:
        if c not in df.columns:
            raise KeyError(f"Column not found: {c}")

    features = read_features(features_path)
    if not features:
        raise ValueError("No feature names found in features file.")

    core = pd.DataFrame(
        {
            "target": df[args.target_col].map(extract_numeric),
            "generated": df[args.generated_col].map(extract_numeric),
            "baseline": df[args.baseline_col].map(extract_numeric),
        }
    )
    valid_core = core.dropna(subset=["target", "generated", "baseline"]).index
    if len(valid_core) == 0:
        raise ValueError("No valid rows after cleaning target/generated/baseline.")

    dfx = df.loc[valid_core].copy()
    dfx["res_generated"] = core.loc[valid_core, "generated"] - core.loc[valid_core, "target"]
    dfx["res_baseline"] = core.loc[valid_core, "baseline"] - core.loc[valid_core, "target"]

    made = 0
    skipped = []
    for feat in features:
        if feat not in dfx.columns:
            skipped.append(f"{feat} (missing)")
            continue

        s = dfx[feat]
        s_num = s.map(extract_numeric)
        non_null = s.notna().sum()
        numeric_like_ratio = (s_num.notna().sum() / non_null) if non_null > 0 else 0.0

        group_col = "__grp__"
        if numeric_like_ratio >= args.numeric_like_threshold:
            tmp = s_num
            tmp_non_na = tmp.dropna()
            if tmp_non_na.nunique() < 2:
                skipped.append(f"{feat} (numeric but constant)")
                continue
            bins = min(args.max_bins, int(tmp_non_na.nunique()))
            try:
                dfx[group_col] = pd.qcut(tmp, q=bins, duplicates="drop").astype("string")
            except Exception:
                dfx[group_col] = pd.cut(tmp, bins=bins, duplicates="drop").astype("string")
        else:
            ss = s.astype("string").fillna("__MISSING__")
            top = ss.value_counts().head(args.max_bins).index
            dfx[group_col] = np.where(ss.isin(top), ss, "__OTHER__")
            dfx[group_col] = pd.Series(dfx[group_col], index=dfx.index, dtype="string")

        groups = dfx[group_col].fillna("__MISSING__").astype("string")
        group_values = groups.value_counts().index.tolist()
        n_panels = len(group_values)
        if n_panels == 0:
            skipped.append(f"{feat} (no groups)")
            continue

        cols = min(3, n_panels)
        rows = int(math.ceil(n_panels / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.8 * rows), squeeze=False)

        lo = np.nanquantile(np.concatenate([dfx["res_generated"].to_numpy(), dfx["res_baseline"].to_numpy()]), 0.01)
        hi = np.nanquantile(np.concatenate([dfx["res_generated"].to_numpy(), dfx["res_baseline"].to_numpy()]), 0.99)
        bins = np.linspace(lo, hi, 70)

        for i, g in enumerate(group_values):
            r = i // cols
            c = i % cols
            ax = axes[r][c]
            m = groups == g
            if int(m.sum()) == 0:
                ax.set_visible(False)
                continue
            rg = dfx.loc[m, "res_generated"].to_numpy(dtype=float)
            rb = dfx.loc[m, "res_baseline"].to_numpy(dtype=float)
            ax.hist(rb, bins=bins, alpha=0.45, label="baseline")
            ax.hist(rg, bins=bins, alpha=0.45, label="generated")
            ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
            mae_g = float(np.mean(np.abs(rg)))
            mae_b = float(np.mean(np.abs(rb)))
            ax.set_title(f"{g}\n n={int(m.sum())}, MAE gen/base={mae_g:.2f}/{mae_b:.2f}")
            ax.set_xlabel("residual = pred - target")
            ax.set_ylabel("count")
            ax.legend()

        for j in range(n_panels, rows * cols):
            axes[j // cols][j % cols].set_visible(False)

        fig.suptitle(f"Residual histograms conditioned on feature: {feat}")
        fig.tight_layout()
        out_png = out_dir / f"{sanitize_filename(feat)}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        made += 1

    print(f"Saved feature-conditional residual plots: {made}")
    if skipped:
        print("Skipped:")
        for s in skipped:
            print(f"- {s}")
    print(f"Output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

