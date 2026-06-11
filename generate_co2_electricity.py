#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CO₂ (electricity) data cleaning pipeline
========================================

Reads raw train/test CSV, applies row and column cleaning, and writes
task-specific cleaned CSV files. No model training or prediction.

Cleaning steps
--------------
1. Coerce target column to numeric.
2. Drop training rows with missing or non-positive target.
3. Remove leakage columns.
4. Drop sparse, constant, or id-like columns (same rules as the old
   ``prepare_features`` step, without ML encoding/imputation).

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
OUTPUT_TRAIN = DATA_DIR / "train_co2_electricity_clean.csv"
OUTPUT_TEST = DATA_DIR / "test_co2_electricity_clean.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "clean_summary_co2_electricity.json"

TARGET_COL = "esg_firma_environmental__co2-ausstoss-elektrizitaet__wert"
COMPANY_COL = "dfo-firmen-daten__steuerungs-daten__crefonummer"
YEAR_COL = "bezugsjahr"

DROP_ZERO_TARGET = True

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


def _is_id_like(col: str) -> bool:
    return any(t in col.lower() for t in _ID_TOKENS)


def select_usable_columns(
    df: pd.DataFrame,
    exclude_cols: Set[str],
    essential_cols: Set[str],
) -> List[str]:
    """Return column order for cleaned CSV: essentials first, then usable features."""
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


def clean_co2_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Apply row/column cleaning and return (train_clean, test_clean, summary)."""
    n_train_raw = len(train_df)
    n_test_raw = len(test_df)
    cols_train_raw = len(train_df.columns)
    cols_test_raw = len(test_df.columns)

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[TARGET_COL] = coerce_numeric(train_df[TARGET_COL])
    if TARGET_COL in test_df.columns:
        test_df[TARGET_COL] = coerce_numeric(test_df[TARGET_COL])

    keep = train_df[TARGET_COL].notna()
    keep &= (train_df[TARGET_COL] > 0) if DROP_ZERO_TARGET else (train_df[TARGET_COL] >= 0)
    train_df = train_df.loc[keep].copy().reset_index(drop=True)

    leakage_present = [c for c in LEAKAGE_FEATURES if c in train_df.columns or c in test_df.columns]
    train_df = train_df.drop(columns=[c for c in leakage_present if c in train_df.columns], errors="ignore")
    test_df = test_df.drop(columns=[c for c in leakage_present if c in test_df.columns], errors="ignore")

    essential = {TARGET_COL, COMPANY_COL, YEAR_COL}
    exclude = {TARGET_COL}
    keep_cols = select_usable_columns(train_df, exclude_cols=exclude, essential_cols=essential)
    keep_cols = [c for c in keep_cols if c in train_df.columns or c in test_df.columns]

    train_out = train_df.reindex(columns=keep_cols).copy()
    test_out = test_df.reindex(columns=keep_cols).copy()

    summary: Dict[str, object] = {
        "task": "co2_electricity",
        "target_col": TARGET_COL,
        "train_input_rows": n_train_raw,
        "test_input_rows": n_test_raw,
        "train_output_rows": len(train_out),
        "test_output_rows": len(test_out),
        "train_rows_dropped": n_train_raw - len(train_out),
        "train_input_cols": cols_train_raw,
        "test_input_cols": cols_test_raw,
        "output_cols": len(keep_cols),
        "leakage_cols_dropped": leakage_present,
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

    train_clean, test_clean, summary = clean_co2_data(train_df, test_df)

    train_clean.to_csv(OUTPUT_TRAIN, index=False, encoding="utf-8-sig")
    test_clean.to_csv(OUTPUT_TEST, index=False, encoding="utf-8-sig")

    summary["output_files"] = {
        "train": str(OUTPUT_TRAIN),
        "test": str(OUTPUT_TEST),
    }
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== CO₂ electricity data cleaning summary ===")
    print(f"Train rows: {summary['train_input_rows']} -> {summary['train_output_rows']}")
    print(f"Test rows : {summary['test_input_rows']} -> {summary['test_output_rows']}")
    print(f"Columns   : {summary['train_input_cols']} -> {summary['output_cols']}")
    print(f"Leakage columns removed: {len(summary['leakage_cols_dropped'])}")
    print(f"\nSaved train: {OUTPUT_TRAIN}")
    print(f"Saved test : {OUTPUT_TEST}")
    print(f"Saved summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
