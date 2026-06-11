# -*- coding: utf-8 -*-
"""
Water consumption data cleaning pipeline
========================================

Reads raw train/test CSV, applies row and column cleaning, and writes
task-specific cleaned CSV files. No model training or prediction.

Cleaning steps
--------------
1. Deduplicate by (company, year) keeping best source priority.
2. Coerce numeric columns; normalize branch code and employee size bin.
3. Drop training rows with missing or non-positive target/baseline.
4. Hierarchical baseline calibration (branch × size_bin, Bayesian shrinkage).
5. Add derived baseline/log columns used downstream.
6. Remove leakage columns.
7. Drop sparse, constant, or id-like columns.

Requirements
------------
pip install pandas numpy
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR
OUTPUT_DIR = ROOT_DIR / "outputs"

TRAIN_PATH = DATA_DIR / "train_clean.csv"
TEST_PATH = DATA_DIR / "test_clean.csv"
OUTPUT_TRAIN = DATA_DIR / "train_water_clean.csv"
OUTPUT_TEST = DATA_DIR / "test_water_clean.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "clean_summary_water.json"

TARGET_COL = "esg_firma_esg-bewertung__input__wasserverbrauch-m3"
BASELINE_COL = "esg_firma_wasser_berechnet"
EMPLOYEE_COL = "esg_firma_esg-bewertung__input__anzahl-mitarbeiter"
COMPANY_COL = "dfo-firmen-daten__steuerungs-daten__crefonummer"
YEAR_COL = "bezugsjahr"
SOURCE_COL = "esg_firma_environmental__wasserverbrauch__quelle"
BRANCH2_COL = "esg_firma_branchencode-2-stellig"

DROP_ZERO_TARGET = True

SOURCE_PRIORITY_MAP_NUMERIC = {
    5: 1, 6: 2, 3: 3, 4: 4, 7: 5, 8: 6, 2: 7, 1: 8, 10: 9, 11: 10, 9: 11,
}
SOURCE_PRIORITY_MAP_TEXT = {
    "nachhaltigkeitsbericht unternehmen": 1,
    "sustainability report (company)": 1,
    "nachhaltigkeitsbericht konzern": 2,
    "sustainability report (group)": 2,
    "jahresbericht unternehmen": 3,
    "annual report (company)": 3,
    "jahresbericht konzern": 4,
    "annual report (group)": 4,
    "myesg-fragebogen (basic)": 5,
    "myesg questionnaire (basic)": 5,
    "myesg-fragebogen (advanced)": 6,
    "myesg questionnaire (advanced)": 6,
    "persönliches interview": 7,
    "personal interview": 7,
    "homepage unternehmen": 8,
    "company homepage": 8,
    "automatische vorbelegung archiv": 9,
    "automatic pre-fill archive": 9,
    "extrapolation": 10,
    "automatische vorbelegung": 11,
    "automatic pre-fill": 11,
}

PRIOR_BRANCH = 40.0
PRIOR_BRANCH_SIZE = 20.0

LEAKAGE_FEATURES = {
    "esg_firma_esg-bewertung__bewertung__environmental__wasserverbrauch-pro-kopf",
    "esg_firma_environmental__wasserverbrauch__wert",
    "esg_firma_Wasser pro Mitarbeiter",
    "esg_firma_firma-esg-score-2__environmental__wasserverbrauch",
}

DERIVED_COLS = [
    "size_bin",
    "source_priority",
    "baseline_calibrated",
    "log_baseline",
    "log_baseline_calibrated",
    "log_employees",
    "calib_shift",
    "log_baseline_per_emp",
]

_ID_TOKENS = {"id", "uuid", "guid", "cref", "nummer", "nr"}


def coerce_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    x = (
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(r"[^\d\.\-eE+]", "", regex=True)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "none": np.nan})
    )
    return pd.to_numeric(x, errors="coerce")


def normalize_branch2(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.extract(r"(\d{1,2})", expand=False)
    return x.fillna("UNK").str.zfill(2)


def make_size_bin(emp: pd.Series) -> pd.Series:
    x = coerce_numeric(emp)
    bins = pd.cut(
        x,
        bins=[-np.inf, 10, 50, 250, 1000, np.inf],
        labels=["0-10", "11-50", "51-250", "251-1000", "1000+"],
        right=True,
    )
    return bins.astype("object").fillna("UNK")


def source_priority_value(v) -> int:
    if pd.isna(v):
        return 999
    try:
        return SOURCE_PRIORITY_MAP_NUMERIC.get(int(float(v)), 999)
    except Exception:
        pass
    return SOURCE_PRIORITY_MAP_TEXT.get(str(v).strip().lower(), 999)


def add_source_priority(df: pd.DataFrame, source_col: str) -> pd.Series:
    if source_col not in df.columns:
        return pd.Series(999, index=df.index)
    return df[source_col].map(source_priority_value).fillna(999).astype(int)


def completeness_score(df: pd.DataFrame, important_cols: List[str]) -> pd.Series:
    valid = [c for c in important_cols if c in df.columns]
    return df[valid].notna().sum(axis=1) if valid else pd.Series(0, index=df.index)


def deduplicate_best_entry(
    df: pd.DataFrame,
    company_col: str,
    year_col: str,
    source_col: str,
    important_cols: List[str],
) -> pd.DataFrame:
    out = df.copy()
    out["_prio"] = add_source_priority(out, source_col)
    out["_comp"] = completeness_score(out, important_cols)

    key_cols = [c for c in [company_col, year_col] if c and c in out.columns]
    if not key_cols:
        return out.drop(columns=["_prio", "_comp"], errors="ignore")

    out = (
        out.sort_values(key_cols + ["_prio", "_comp"], ascending=[True] * len(key_cols) + [True, False])
        .drop_duplicates(subset=key_cols, keep="first")
        .drop(columns=["_prio", "_comp"], errors="ignore")
    )
    return out


def winsorize_series(s: pd.Series, q_low=0.01, q_high=0.99) -> pd.Series:
    return s.clip(s.quantile(q_low), s.quantile(q_high))


def build_hierarchical_calibrator(
    df_train: pd.DataFrame,
    target_col: str,
    baseline_col: str,
    branch2_col: str,
    size_bin_col: str,
) -> dict:
    tmp = df_train[[target_col, baseline_col, branch2_col, size_bin_col]].copy()
    tmp["lr"] = winsorize_series(
        np.log1p(tmp[target_col]) - np.log1p(tmp[baseline_col]), 0.01, 0.99
    )
    global_mean = float(tmp["lr"].mean())

    branch_tbl = (
        tmp.groupby(branch2_col)["lr"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "b_mean", "count": "b_cnt"})
    )

    seg_tbl = (
        tmp.groupby([branch2_col, size_bin_col])["lr"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "s_mean", "count": "s_cnt"})
    )

    return {
        "global_mean": global_mean,
        "branch_tbl": branch_tbl,
        "seg_tbl": seg_tbl,
        "branch2_col": branch2_col,
        "size_bin_col": size_bin_col,
    }


def apply_hierarchical_calibrator(
    df: pd.DataFrame, calibrator: dict, baseline_col: str
) -> pd.Series:
    b2 = calibrator["branch2_col"]
    sb = calibrator["size_bin_col"]
    gm = calibrator["global_mean"]

    out = df.reset_index(drop=True).copy()
    out = out.merge(calibrator["branch_tbl"], how="left", on=b2)
    out = out.merge(calibrator["seg_tbl"], how="left", on=[b2, sb])

    b_mean = out["b_mean"].fillna(gm)
    b_cnt = out["b_cnt"].fillna(0.0)
    adj_b = (b_mean * b_cnt + PRIOR_BRANCH * gm) / (b_cnt + PRIOR_BRANCH)

    s_mean = out["s_mean"].fillna(adj_b)
    s_cnt = out["s_cnt"].fillna(0.0)
    adj_s = (s_mean * s_cnt + PRIOR_BRANCH_SIZE * adj_b) / (s_cnt + PRIOR_BRANCH_SIZE)

    base = coerce_numeric(out[baseline_col]).clip(lower=0)
    cal = np.maximum(np.expm1(np.log1p(base) + adj_s), 0)

    return pd.Series(cal.values, index=df.index, name="baseline_calibrated")


def _is_id_like(col: str) -> bool:
    return any(t in col.lower() for t in _ID_TOKENS)


def select_usable_columns(
    df: pd.DataFrame,
    exclude_cols: Set[str],
    essential_cols: Set[str],
) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()

    for col in df.columns:
        if col in seen:
            continue
        if col in essential_cols:
            ordered.append(col)
            seen.add(col)

    for col in df.columns:
        if col in seen or col in exclude_cols:
            continue
        na_r = df[col].isna().mean()
        nu = df[col].nunique(dropna=True)
        if na_r >= 0.995 or nu <= 1:
            continue
        if nu / max(1, df[col].notna().sum()) > 0.98 and _is_id_like(col):
            continue
        ordered.append(col)
        seen.add(col)

    return ordered


def add_derived_columns(df: pd.DataFrame) -> None:
    lb = np.log1p(df[BASELINE_COL].clip(lower=0))
    lbc = np.log1p(df["baseline_calibrated"].clip(lower=0))
    le = np.log1p(df[EMPLOYEE_COL].clip(lower=0))
    df["log_baseline"] = lb
    df["log_baseline_calibrated"] = lbc
    df["log_employees"] = le
    df["calib_shift"] = lbc - lb
    emp_safe = df[EMPLOYEE_COL].clip(lower=1).fillna(1)
    df["log_baseline_per_emp"] = np.log1p(df[BASELINE_COL].clip(lower=0) / emp_safe)


def clean_water_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Apply row/column cleaning and return (train_clean, test_clean, summary)."""
    n_train_raw = len(train_df)
    n_test_raw = len(test_df)
    cols_train_raw = len(train_df.columns)

    important_cols = [TARGET_COL, BASELINE_COL, EMPLOYEE_COL, BRANCH2_COL]
    train_df = deduplicate_best_entry(train_df, COMPANY_COL, YEAR_COL, SOURCE_COL, important_cols)
    test_df = deduplicate_best_entry(test_df, COMPANY_COL, YEAR_COL, SOURCE_COL, important_cols)
    n_train_after_dedup = len(train_df)
    n_test_after_dedup = len(test_df)

    for df in [train_df, test_df]:
        df[BASELINE_COL] = coerce_numeric(df[BASELINE_COL])
        if EMPLOYEE_COL in df.columns:
            df[EMPLOYEE_COL] = coerce_numeric(df[EMPLOYEE_COL])
        df[BRANCH2_COL] = normalize_branch2(df[BRANCH2_COL])
        df["size_bin"] = make_size_bin(df[EMPLOYEE_COL])
        df["source_priority"] = add_source_priority(df, SOURCE_COL)
        if YEAR_COL in df.columns:
            df[YEAR_COL] = coerce_numeric(df[YEAR_COL])

    train_df[TARGET_COL] = coerce_numeric(train_df[TARGET_COL])
    if TARGET_COL in test_df.columns:
        test_df[TARGET_COL] = coerce_numeric(test_df[TARGET_COL])

    keep = (
        train_df[TARGET_COL].notna()
        & train_df[BASELINE_COL].notna()
        & (train_df[BASELINE_COL] > 0)
    )
    keep &= (train_df[TARGET_COL] > 0) if DROP_ZERO_TARGET else (train_df[TARGET_COL] >= 0)
    train_df = train_df.loc[keep].copy().reset_index(drop=True)

    calibrator = build_hierarchical_calibrator(
        train_df,
        target_col=TARGET_COL,
        baseline_col=BASELINE_COL,
        branch2_col=BRANCH2_COL,
        size_bin_col="size_bin",
    )
    train_df["baseline_calibrated"] = apply_hierarchical_calibrator(train_df, calibrator, BASELINE_COL)
    test_df["baseline_calibrated"] = apply_hierarchical_calibrator(test_df, calibrator, BASELINE_COL)

    for df in [train_df, test_df]:
        add_derived_columns(df)

    leakage_present = [c for c in LEAKAGE_FEATURES if c in train_df.columns or c in test_df.columns]
    train_df = train_df.drop(columns=[c for c in leakage_present if c in train_df.columns], errors="ignore")
    test_df = test_df.drop(columns=[c for c in leakage_present if c in test_df.columns], errors="ignore")

    essential = {
        TARGET_COL,
        BASELINE_COL,
        EMPLOYEE_COL,
        COMPANY_COL,
        YEAR_COL,
        SOURCE_COL,
        BRANCH2_COL,
        *DERIVED_COLS,
    }
    exclude: Set[str] = set()
    keep_cols = select_usable_columns(train_df, exclude_cols=exclude, essential_cols=essential)
    keep_cols = [c for c in keep_cols if c in train_df.columns or c in test_df.columns]

    train_out = train_df.reindex(columns=keep_cols).copy()
    test_out = test_df.reindex(columns=keep_cols).copy()

    summary: Dict[str, object] = {
        "task": "water",
        "target_col": TARGET_COL,
        "baseline_col": BASELINE_COL,
        "train_input_rows": n_train_raw,
        "test_input_rows": n_test_raw,
        "train_rows_after_dedup": n_train_after_dedup,
        "test_rows_after_dedup": n_test_after_dedup,
        "train_output_rows": len(train_out),
        "test_output_rows": len(test_out),
        "train_rows_dropped_after_dedup": n_train_after_dedup - len(train_out),
        "train_input_cols": cols_train_raw,
        "output_cols": len(keep_cols),
        "leakage_cols_dropped": leakage_present,
        "derived_cols_added": DERIVED_COLS,
        "output_columns": keep_cols,
    }
    return train_out, test_out, summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading train: {TRAIN_PATH}")
    print(f"Loading test : {TEST_PATH}")
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    test_df = pd.read_csv(TEST_PATH, low_memory=False)

    train_clean, test_clean, summary = clean_water_data(train_df, test_df)

    train_clean.to_csv(OUTPUT_TRAIN, index=False, encoding="utf-8-sig")
    test_clean.to_csv(OUTPUT_TEST, index=False, encoding="utf-8-sig")

    summary["output_files"] = {
        "train": str(OUTPUT_TRAIN),
        "test": str(OUTPUT_TEST),
    }
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Water data cleaning summary ===")
    print(f"Train rows: {summary['train_input_rows']} -> {summary['train_output_rows']}")
    print(f"  (after dedup: {summary['train_rows_after_dedup']})")
    print(f"Test rows : {summary['test_input_rows']} -> {summary['test_output_rows']}")
    print(f"  (after dedup: {summary['test_rows_after_dedup']})")
    print(f"Columns   : {summary['train_input_cols']} -> {summary['output_cols']}")
    print(f"Leakage columns removed: {len(summary['leakage_cols_dropped'])}")
    print(f"Derived columns added: {', '.join(DERIVED_COLS)}")
    print(f"\nSaved train: {OUTPUT_TRAIN}")
    print(f"Saved test : {OUTPUT_TEST}")
    print(f"Saved summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
