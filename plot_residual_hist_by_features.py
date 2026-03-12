#!/usr/bin/env python3
"""
Plot feature-bin vs average error (generated vs baseline).

For each selected feature:
- x-axis: feature bins/categories
- y-axis: mean error vs target (absolute or signed)
- Compare generated and baseline on the same chart
"""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Plot feature-bin vs average error (generated vs baseline).")
    parser.add_argument("--file", default=DEFAULT_FILE)
    parser.add_argument("--features-file", default=DEFAULT_FEATURES_FILE)
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    parser.add_argument("--generated-col", default=DEFAULT_GENERATED_COL)
    parser.add_argument("--baseline-col", default=DEFAULT_BASELINE_COL)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-bins", type=int, default=6, help="Max numeric bins / top categories per feature.")
    parser.add_argument("--numeric-like-threshold", type=float, default=0.75)
    parser.add_argument(
        "--error-type",
        choices=["absolute", "signed"],
        default="absolute",
        help="Error definition: absolute=mean(|pred-target|), signed=mean(pred-target).",
    )
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
        group_order: List[str] = []
        if numeric_like_ratio >= args.numeric_like_threshold:
            tmp = s_num
            tmp_non_na = tmp.dropna()
            if tmp_non_na.nunique() < 2:
                skipped.append(f"{feat} (numeric but constant)")
                continue
            bins = min(args.max_bins, int(tmp_non_na.nunique()))
            try:
                grp = pd.qcut(tmp, q=bins, duplicates="drop")
            except Exception:
                grp = pd.cut(tmp, bins=bins, duplicates="drop")
            grp = grp.astype("string")
            dfx[group_col] = grp
            group_order = [str(v) for v in pd.Series(grp.dropna().unique()).tolist()]
        else:
            ss = s.astype("string").fillna("__MISSING__")
            top = ss.value_counts().head(args.max_bins).index
            dfx[group_col] = np.where(ss.isin(top), ss, "__OTHER__")
            dfx[group_col] = pd.Series(dfx[group_col], index=dfx.index, dtype="string")
            group_order = list(top.astype(str)) + ["__OTHER__"]

        groups = dfx[group_col].fillna("__MISSING__").astype("string")
        if not group_order:
            group_order = groups.value_counts().index.astype(str).tolist()
        if "__MISSING__" in set(groups.astype(str)) and "__MISSING__" not in group_order:
            group_order.append("__MISSING__")

        if len(group_order) == 0:
            skipped.append(f"{feat} (no groups)")
            continue

        x_labels: List[str] = []
        gen_mean_err: List[float] = []
        base_mean_err: List[float] = []
        for g in group_order:
            m = groups.astype(str) == g
            n = int(m.sum())
            if n == 0:
                continue
            rg = dfx.loc[m, "res_generated"].to_numpy(dtype=float)
            rb = dfx.loc[m, "res_baseline"].to_numpy(dtype=float)
            if args.error_type == "absolute":
                gen_v = float(np.mean(np.abs(rg)))
                base_v = float(np.mean(np.abs(rb)))
            else:
                gen_v = float(np.mean(rg))
                base_v = float(np.mean(rb))
            x_labels.append(f"{g}\n(n={n})")
            gen_mean_err.append(gen_v)
            base_mean_err.append(base_v)

        if not x_labels:
            skipped.append(f"{feat} (all groups empty)")
            continue

        x = np.arange(len(x_labels))
        w = 0.38
        fig_w = max(10, len(x_labels) * 1.7)
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, 5.5))
        ax.bar(x - w / 2, base_mean_err, width=w, label="baseline", alpha=0.7)
        ax.bar(x + w / 2, gen_mean_err, width=w, label="generated", alpha=0.7)
        if args.error_type == "signed":
            ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=25, ha="right")
        ax.set_xlabel(f"{feat} (binned/grouped)")
        ax.set_ylabel(
            "mean absolute error"
            if args.error_type == "absolute"
            else "mean signed error (pred-target)"
        )
        ax.set_title(f"Feature-conditioned average error: {feat}")
        ax.legend()
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

