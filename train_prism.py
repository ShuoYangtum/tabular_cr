#!/usr/bin/env python3
"""
PRISM: Prior-anchored Robust quantile modeling with conformal trust Intervals
for Skewed Measurements.

A heavy-tail-aware regression pipeline for dirty ESG tabular data
(water / electricity consumption prediction).

Key ideas
---------
1. Median-optimal robust core:
   All modeling happens in z = log1p(target) space with *quantile* loss.
   The conditional median is invariant under monotone transforms, so the
   q50 model in log space is (approximately) the MAE-optimal point
   predictor in the original space -- without letting the heavy tail
   dominate the loss.

2. Unified anchor (works with or without a domain baseline):
   - If a rule-based baseline column exists (water), it is used both as a
     feature and as the "anchor" that predictions are conservatively
     shrunk toward.
   - If not (electricity), PRISM synthesizes its own anchor: a
     hierarchical empirical-Bayes prior (shrunk group medians over
     automatically selected categorical columns), computed out-of-fold to
     stay leak-free.

3. Tail-risk head:
   A classifier estimates P(sample is in the top target decile). For
   high-risk samples only, the point estimate switches from the median to
   an upper quantile. This directly targets the observed failure mode
   where >90% of MAE concentrates in the top ~5% of samples.

4. Conservative shrinkage with a median guard:
   The blend  z_hat = anchor + lambda * (z_model - anchor)  is tuned
   out-of-fold. Candidates whose validation MedianAE is worse than the
   anchor's by more than a small tolerance are rejected outright, and if
   nothing beats the anchor the pipeline falls back to it. This prevents
   "improve RMSE, ruin the median" regressions.

5. Conformalized quantile intervals (CQR) + trust score:
   Every prediction ships with a calibrated interval and a trust score in
   [0, 1], so downstream users know which predictions to rely on.

Usage
-----
Water (with rule-based baseline):
  python train_prism.py --train-file train_clean.csv --test-file test_clean.csv \
      --target-col esg_firma_esg-bewertung__input__wasserverbrauch-m3 \
      --baseline-col esg_firma_wasser_berechnet

Electricity (no baseline -> auto-synthesized prior anchor):
  python train_prism.py --train-file train_clean.csv --test-file test_clean.csv \
      --target-col esg_firma_esg-bewertung__input__elektrizitaetsverbrauch-kwh
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# =========================
# Defaults
# =========================
DEFAULT_TRAIN_FILE = "train_clean.csv"
DEFAULT_TEST_FILE = "test_clean.csv"
DEFAULT_OUTPUT_FILE = "generated.csv"
DEFAULT_TARGET_COL = "esg_firma_esg-bewertung__input__elektrizitaetsverbrauch-kwh"
DEFAULT_BASELINE_COL = ""  # empty -> synthesized prior anchor
DEFAULT_GENERATED_COL = "generated"
DEFAULT_REPORT_FILE = "prism_report.json"
DEFAULT_OOF_FILE = "prism_oof.csv"

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
DEFAULT_LEAKAGE_FEATURES = ",".join(sorted(LEAKAGE_FEATURES))

QUANTILES = (0.05, 0.5, 0.75, 0.9, 0.95)
INTERVAL = (0.05, 0.95)  # nominal 90% interval, conformally calibrated
LAMBDA_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
TAIL_TAU_GRID = (1.1, 0.5, 0.6, 0.7, 0.8)  # 1.1 == "never switch"
TAIL_QHI_GRID = (0.75, 0.9)

NUMERIC_LIKE_THRESHOLD = 0.85
NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
MISSING_STRINGS = {"", "nan", "none", "null", "na", "n/a", "-", "--", "unknown"}
FLOAT32_SAFE_MAX = np.finfo(np.float32).max * 0.99
EPS = 1e-12


# =========================
# Value cleaning
# =========================
def extract_numeric(value) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        v = float(value)
        if math.isnan(v) or math.isinf(v) or abs(v) > FLOAT32_SAFE_MAX:
            return np.nan
        return v
    s = str(value).strip()
    if not s:
        return np.nan
    s = re.sub(r"\s+", "", s.replace(",", ""))
    if s.lower() in MISSING_STRINGS:
        return np.nan
    m = NUMERIC_PATTERN.search(s)
    if not m:
        return np.nan
    try:
        v = float(m.group())
        if math.isnan(v) or math.isinf(v) or abs(v) > FLOAT32_SAFE_MAX:
            return np.nan
        return v
    except (ValueError, OverflowError):
        return np.nan


def numeric_like(value) -> bool:
    """True if the value should count as numeric for *type inference*.

    Stricter than extract_numeric: a category label like "branche_3" contains a
    digit but is not a number, so we also require that numeric characters make
    up a reasonable fraction of the string.
    """
    if value is None:
        return False
    if isinstance(value, (int, float, np.number)):
        return not (isinstance(value, float) and math.isnan(value))
    s = str(value).strip()
    if not s or s.lower() in MISSING_STRINGS:
        return False
    if math.isnan(extract_numeric(s)):
        return False
    compact = re.sub(r"\s+", "", s)
    digit_frac = sum(ch.isdigit() or ch in ".,+-" for ch in compact) / max(len(compact), 1)
    return digit_frac >= 0.35


def clean_categorical(value):
    if value is None:
        return np.nan
    s = str(value).strip()
    if s.lower() in MISSING_STRINGS:
        return np.nan
    return s


def parse_datetime_series(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not infer format.*", category=UserWarning)
        try:
            return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
        except TypeError:
            return pd.to_datetime(series, errors="coerce", utc=True)


# =========================
# Semantic feature engineering (pure heuristics; no LLM needed)
# =========================
ID_NAME_HINTS = ("id", "uuid", "guid", "key", "nummer", "crefonummer", "timestamp", "zeitpunkt")


def column_profile(s: pd.Series) -> Dict[str, float]:
    total = int(s.shape[0])
    non_na = max(int(s.notna().sum()), 1)
    numeric = pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")
    dt = parse_datetime_series(s)
    prof = {
        "non_null_ratio": float(s.notna().mean()) if total else 0.0,
        "unique_ratio": float(s.nunique(dropna=True) / non_na),
        "numeric_ratio": float(numeric.notna().mean()) if total else 0.0,
        "datetime_ratio": float(dt.notna().mean()) if total else 0.0,
        "year_like_ratio": 0.0,
    }
    n = numeric.dropna()
    if len(n):
        prof["year_like_ratio"] = float(((n >= 1900) & (n <= 2100) & (np.floor(n) == n)).mean())
    return prof


def engineer_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, int]]:
    """Type each column heuristically and expand datetimes / years, drop IDs."""
    out_cols: List[str] = []
    tr_new: Dict[str, pd.Series] = {}
    te_new: Dict[str, pd.Series] = {}
    summary = {"dropped_id_like": 0, "datetime_expanded": 0, "year_transformed": 0}

    for col in feature_cols:
        s = train_df[col]
        prof = column_profile(s)
        low = col.lower()
        is_id = (
            any(h in low for h in ID_NAME_HINTS)
            and prof["unique_ratio"] >= 0.95
            and s.notna().sum() >= 200
        )
        if is_id:
            summary["dropped_id_like"] += 1
            continue
        if prof["datetime_ratio"] >= 0.8 and prof["numeric_ratio"] < 0.8:
            summary["datetime_expanded"] += 1
            for name, df_src, sink in (("tr", train_df, tr_new), ("te", test_df, te_new)):
                dt = parse_datetime_series(df_src[col]) if col in df_src.columns else pd.Series(pd.NaT, index=df_src.index)
                sink[f"{col}__year"] = dt.dt.year
                sink[f"{col}__month"] = dt.dt.month
                sink[f"{col}__weekday"] = dt.dt.weekday
            out_cols += [f"{col}__year", f"{col}__month", f"{col}__weekday"]
            continue
        if prof["numeric_ratio"] >= 0.8 and prof["year_like_ratio"] >= 0.8:
            summary["year_transformed"] += 1
            for df_src, sink in ((train_df, tr_new), (test_df, te_new)):
                yr = df_src[col].map(extract_numeric) if col in df_src.columns else pd.Series(np.nan, index=df_src.index)
                sink[f"{col}__age"] = pd.Timestamp.now().year - yr
            out_cols.append(f"{col}__age")
            continue
        out_cols.append(col)

    if tr_new:
        train_df = pd.concat([train_df, pd.DataFrame(tr_new, index=train_df.index)], axis=1)
    if te_new:
        test_df = pd.concat([test_df, pd.DataFrame(te_new, index=test_df.index)], axis=1)
    return train_df.copy(), test_df.copy(), out_cols, summary


def infer_feature_types(df: pd.DataFrame, threshold: float) -> Tuple[List[str], List[str]]:
    numeric_cols, categorical_cols = [], []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
            continue
        non_null = s.notna().sum()
        if non_null == 0:
            categorical_cols.append(col)
            continue
        if s.map(numeric_like).sum() / non_null >= threshold:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def apply_feature_cleaning(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    cleaned = {}
    for col in numeric_cols:
        cleaned[col] = df[col].map(extract_numeric).replace([np.inf, -np.inf], np.nan)
    for col in categorical_cols:
        cleaned[col] = df[col].map(clean_categorical)
    return pd.DataFrame(cleaned, index=df.index)


def drop_sparse_or_constant(
    x_train: pd.DataFrame, x_test: pd.DataFrame, min_non_null_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    keep = []
    dropped_sparse = dropped_const = 0
    for col in x_train.columns:
        if float(x_train[col].notna().mean()) < min_non_null_ratio:
            dropped_sparse += 1
            continue
        if int(x_train[col].nunique(dropna=True)) <= 1:
            dropped_const += 1
            continue
        keep.append(col)
    return (
        x_train[keep].copy(),
        x_test[keep].copy(),
        {"dropped_sparse": dropped_sparse, "dropped_constant": dropped_const, "kept": len(keep)},
    )


# =========================
# Preprocessor
# =========================
class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, low_q: float = 0.005, high_q: float = 0.995):
        self.low_q = low_q
        self.high_q = high_q

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(arr, self.low_q, axis=0)
        self.upper_ = np.nanquantile(arr, self.high_q, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("clip", QuantileClipper()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric_cols), ("cat", categorical_pipe, categorical_cols)],
        remainder="drop",
    )


def make_quantile_regressor(q: float, n_estimators: int, lr: float, seed: int):
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=q,
        learning_rate=lr,
        max_iter=n_estimators,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1e-3,
        random_state=seed,
    )


def make_classifier(n_estimators: int, lr: float, seed: int):
    return HistGradientBoostingClassifier(
        learning_rate=lr,
        max_iter=n_estimators,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1e-3,
        random_state=seed,
    )


# =========================
# Hierarchical empirical-Bayes prior anchor
# =========================
@dataclass
class EBPrior:
    """Hierarchical shrunk-group-median prior over categorical columns.

    Level 0 is the global median of z; each subsequent level refines it with
    the group median of the key (col_1, ..., col_k), shrunk toward the parent
    prior with pseudo-count m (empirical-Bayes shrinkage). Sparse or unseen
    groups automatically fall back to their parent level.
    """

    cols: List[str]
    m: float = 20.0
    global_median_: float = 0.0
    levels_: List[Dict[str, Tuple[int, float]]] = field(default_factory=list)

    @staticmethod
    def _keys(df: pd.DataFrame, cols: List[str]) -> pd.Series:
        parts = [df[c].astype("string").fillna("__NA__") for c in cols]
        key = parts[0]
        for p in parts[1:]:
            key = key.str.cat(p, sep="\x1f")
        return key

    def fit(self, df: pd.DataFrame, z: np.ndarray) -> "EBPrior":
        z_s = pd.Series(np.asarray(z, dtype=float), index=df.index)
        self.global_median_ = float(z_s.median())
        self.levels_ = []
        for depth in range(1, len(self.cols) + 1):
            keys = self._keys(df, self.cols[:depth])
            grp = z_s.groupby(keys).agg(["count", "median"])
            self.levels_.append({k: (int(r["count"]), float(r["median"])) for k, r in grp.iterrows()})
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        prior = np.full(len(df), self.global_median_, dtype=float)
        for depth, table in enumerate(self.levels_, start=1):
            keys = self._keys(df, self.cols[:depth]).to_numpy()
            for i, k in enumerate(keys):
                stat = table.get(k)
                if stat is None:
                    continue
                n_g, med_g = stat
                prior[i] = (n_g * med_g + self.m * prior[i]) / (n_g + self.m)
        return prior


def select_prior_columns(
    x_train: pd.DataFrame,
    z: np.ndarray,
    categorical_cols: List[str],
    n_levels: int,
    m: float,
    seed: int,
) -> List[Tuple[str, float]]:
    """Score candidate categorical columns by 2-fold cross-fitted MAE reduction
    of the shrunk-group-median prior over the global median; return top columns."""
    n = len(z)
    z_arr = np.asarray(z, dtype=float)
    rng = np.random.RandomState(seed)
    half = rng.permutation(n) < n // 2
    scores: List[Tuple[str, float]] = []
    for col in categorical_cols:
        s = x_train[col]
        nunique = s.nunique(dropna=True)
        if s.notna().mean() < 0.3 or nunique < 2 or nunique > max(2, n // 4):
            continue
        gain = 0.0
        for fit_mask in (half, ~half):
            prior = EBPrior(cols=[col], m=m).fit(x_train.loc[fit_mask], z_arr[fit_mask])
            pred = prior.predict(x_train.loc[~fit_mask])
            base_mae = float(np.mean(np.abs(z_arr[~fit_mask] - prior.global_median_)))
            col_mae = float(np.mean(np.abs(z_arr[~fit_mask] - pred)))
            gain += (base_mae - col_mae) / 2.0
        scores.append((col, gain))
    scores.sort(key=lambda kv: kv[1], reverse=True)
    return [kv for kv in scores[:n_levels] if kv[1] > 0]


# =========================
# Domain (train->test) validation weights
# =========================
def build_domain_weights(
    x_train_t: np.ndarray, x_test_t: np.ndarray, max_weight: float, seed: int
) -> Tuple[np.ndarray, float]:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    x_all = np.vstack([x_train_t, x_test_t])
    y_all = np.concatenate([np.zeros(len(x_train_t)), np.ones(len(x_test_t))])
    clf = make_classifier(n_estimators=120, lr=0.05, seed=seed)
    # cross-validated probabilities: leak-free weights and an honest AUC
    p_all = cross_val_predict(clf, x_all, y_all, cv=3, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y_all, p_all))
    p = np.clip(p_all[: len(x_train_t)], 1e-4, 1 - 1e-4)
    w = np.clip(p / (1 - p), 1.0 / max_weight, max_weight)
    w = w / max(float(np.mean(w)), EPS)
    return w, auc


# =========================
# Metrics & tuning
# =========================
def metric_table(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray | None = None) -> Dict[str, float]:
    """Unweighted deployment metrics, plus w_-prefixed domain-weighted variants
    of the scale metrics when validation weights are supplied."""
    err = y_pred - y_true
    abs_err = np.abs(err)
    denom = max(float(np.sum(np.abs(y_true))), EPS)
    zt, zp = np.log1p(np.maximum(y_true, 0)), np.log1p(np.maximum(y_pred, 0))
    log_err = np.abs(zp - zt)
    out = {
        "mae": float(np.mean(abs_err)),
        "median_ae": float(np.median(abs_err)),
        "wape": float(np.sum(abs_err) / denom),
        "log_mae": float(np.mean(log_err)),
        "trimmed_mae90": float(np.mean(np.sort(abs_err)[: max(int(len(abs_err) * 0.9), 1)])),
        "bias": float(np.mean(err)),
    }
    if w is not None:
        out["w_mae"] = float(np.average(abs_err, weights=w))
        out["w_log_mae"] = float(np.average(log_err, weights=w))
    return out


def objective_value(metrics: Dict[str, float], objective: str) -> float:
    # prefer the domain-weighted variant of the chosen objective when available
    if f"w_{objective}" in metrics:
        return float(metrics[f"w_{objective}"])
    if objective in metrics:
        return float(metrics[objective])
    raise ValueError(f"unknown objective {objective}")


def compose_point_estimate(
    anchor_z: np.ndarray,
    preds_z: Dict[float, np.ndarray],
    p_tail: np.ndarray,
    lam: float,
    tau: float,
    q_hi: float,
) -> np.ndarray:
    z_sel = preds_z[0.5].copy()
    if tau <= 1.0:
        mask = p_tail >= tau
        z_sel[mask] = np.maximum(z_sel[mask], preds_z[q_hi][mask])
    return anchor_z + lam * (z_sel - anchor_z)


def tune_composition(
    y_true: np.ndarray,
    anchor_z: np.ndarray,
    preds_z: Dict[float, np.ndarray],
    p_tail: np.ndarray,
    w: np.ndarray | None,
    objective: str,
    median_guard_ratio: float,
    mae_guard_ratio: float,
    anchor_is_real_baseline: bool,
    min_rel_gain: float,
    enable_tail_switch: bool,
) -> Dict[str, Any]:
    anchor_pred = np.expm1(anchor_z)
    anchor_metrics = metric_table(y_true, anchor_pred, w)
    anchor_obj = objective_value(anchor_metrics, objective)

    def guards_ok(m: Dict[str, float]) -> bool:
        # never deploy something that materially regresses the metrics the
        # baseline is judged by, regardless of the tuning objective
        if not anchor_is_real_baseline:
            return True
        if median_guard_ratio > 0 and m["median_ae"] > anchor_metrics["median_ae"] * median_guard_ratio:
            return False
        if mae_guard_ratio > 0 and m["mae"] > anchor_metrics["mae"] * mae_guard_ratio:
            return False
        return True

    taus = TAIL_TAU_GRID if enable_tail_switch else (1.1,)
    fallback = {
        "lam": 0.0,
        "tau": 1.1,
        "q_hi": 0.9,
        "objective": anchor_obj,
        "metrics": anchor_metrics,
        "fallback_to_anchor": True,
    }
    best = dict(fallback)
    best_obj = float("inf")
    for lam in LAMBDA_GRID:
        for tau in taus:
            for q_hi in TAIL_QHI_GRID if tau <= 1.0 else (TAIL_QHI_GRID[0],):
                z_pt = compose_point_estimate(anchor_z, preds_z, p_tail, lam, tau, q_hi)
                y_hat = np.expm1(z_pt)
                m = metric_table(y_true, y_hat, w)
                if not guards_ok(m):
                    continue
                obj = objective_value(m, objective)
                if obj < best_obj:
                    best_obj = obj
                    best = {
                        "lam": lam,
                        "tau": tau,
                        "q_hi": q_hi,
                        "objective": obj,
                        "metrics": m,
                        "fallback_to_anchor": False,
                    }
    # conservative fallback: require a minimum relative gain over the anchor
    if anchor_is_real_baseline and best_obj > anchor_obj * (1.0 - min_rel_gain):
        best = dict(fallback)
    best["anchor_metrics"] = anchor_metrics
    return best


def stratified_report(
    y_true: np.ndarray, y_model: np.ndarray, y_anchor: np.ndarray, n_bins: int = 5
) -> List[Dict[str, float]]:
    z = np.log1p(y_true)
    edges = np.unique(np.quantile(z, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(z, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not np.any(m):
            continue
        rows.append(
            {
                "bin": b,
                "n": int(m.sum()),
                "target_min": float(np.expm1(edges[b])),
                "target_max": float(np.expm1(edges[b + 1])),
                "anchor_mae": float(np.mean(np.abs(y_anchor[m] - y_true[m]))),
                "model_mae": float(np.mean(np.abs(y_model[m] - y_true[m]))),
                "model_win_rate": float(
                    np.mean(np.abs(y_model[m] - y_true[m]) < np.abs(y_anchor[m] - y_true[m]))
                ),
            }
        )
    return rows


# =========================
# Main
# =========================
def main() -> int:
    p = argparse.ArgumentParser(description="PRISM: prior-anchored robust quantile regression for skewed tabular targets.")
    p.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    p.add_argument("--test-file", default=DEFAULT_TEST_FILE)
    p.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    p.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    p.add_argument("--baseline-col", default=DEFAULT_BASELINE_COL,
                   help="Rule-based baseline column; leave empty to auto-synthesize a prior anchor.")
    p.add_argument("--generated-col", default=DEFAULT_GENERATED_COL)
    p.add_argument("--leakage-features", default=DEFAULT_LEAKAGE_FEATURES)
    p.add_argument("--report-file", default=DEFAULT_REPORT_FILE)
    p.add_argument("--oof-file", default=DEFAULT_OOF_FILE)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--learning-rate", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--objective", default="log_mae",
                   choices=["mae", "median_ae", "wape", "log_mae", "trimmed_mae90"],
                   help="Tuning objective. Default log_mae: scale-free relative error, robust to the "
                        "heavy tail; absolute-scale regressions are prevented by the guards instead.")
    p.add_argument("--min-rel-gain", type=float, default=0.005,
                   help="Minimum relative OOF gain over the baseline anchor before deploying the model.")
    p.add_argument("--median-guard-ratio", type=float, default=1.05,
                   help="Reject candidates whose OOF MedianAE exceeds baseline MedianAE by this factor (<=0 disables).")
    p.add_argument("--mae-guard-ratio", type=float, default=1.05,
                   help="Reject candidates whose OOF MAE exceeds baseline MAE by this factor (<=0 disables).")
    p.add_argument("--prior-levels", type=int, default=3)
    p.add_argument("--prior-shrinkage", type=float, default=20.0)
    p.add_argument("--min-feature-non-null-ratio", type=float, default=0.01)
    p.add_argument("--tail-quantile", type=float, default=0.9,
                   help="Train-target quantile defining the tail class for the risk head.")
    p.add_argument("--disable-tail-switch", action="store_true")
    p.add_argument("--disable-prior-anchor", action="store_true",
                   help="Do not synthesize the EB prior (falls back to global median as anchor when no baseline).")
    p.add_argument("--disable-domain-weighting", action="store_true")
    p.add_argument("--domain-weight-max", type=float, default=10.0)
    p.add_argument("--interval-alpha", type=float, default=0.1,
                   help="Miscoverage level for the conformal interval (0.1 -> 90% target coverage).")
    args = p.parse_args()

    rng_seed = int(args.seed)
    train_path, test_path = Path(args.train_file), Path(args.test_file)
    if not train_path.exists() or not test_path.exists():
        print(f"[ERROR] train/test file not found: {train_path} / {test_path}")
        return 1

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)
    train_df.columns = [str(c).strip() for c in train_df.columns]
    test_df.columns = [str(c).strip() for c in test_df.columns]
    test_df_out = test_df.copy()

    target_col, baseline_col = args.target_col, args.baseline_col.strip()
    if target_col not in train_df.columns:
        print(f"[ERROR] target column '{target_col}' not in train file.")
        return 1
    has_baseline = bool(baseline_col) and baseline_col in train_df.columns
    if baseline_col and not has_baseline:
        print(f"[WARN] baseline column '{baseline_col}' not found; running in no-baseline mode.")

    # ---- feature set ----
    leakage_roots = {c.strip() for c in args.leakage_features.split(",") if c.strip()}

    def is_leak(col: str) -> bool:
        return col in leakage_roots or any(col.startswith(f"{r}__") for r in leakage_roots)

    feature_cols = [c for c in train_df.columns if c not in {target_col, baseline_col} and not is_leak(c)]
    print(f"[INFO] {len(feature_cols)} raw feature columns "
          f"({len(train_df.columns) - len(feature_cols)} excluded as target/baseline/leakage).")

    train_df, test_df, feature_cols, eng_summary = engineer_features(train_df, test_df, feature_cols)
    feature_cols = [c for c in feature_cols if not is_leak(c)]
    print(f"[INFO] semantic engineering: {eng_summary}")

    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = np.nan

    # ---- targets / anchor raw values ----
    target_all = train_df[target_col].map(extract_numeric)
    valid_mask = target_all.notna() & (target_all > EPS)
    n_dropped = int((~valid_mask).sum())
    print(f"[INFO] train rows: {len(train_df)} total, {n_dropped} dropped (missing/non-positive target).")

    X_train_raw = train_df.loc[valid_mask, feature_cols].reset_index(drop=True)
    y = target_all.loc[valid_mask].to_numpy(dtype=float)
    z = np.log1p(y)
    n = len(y)
    if n < 50:
        print("[ERROR] too few valid training rows.")
        return 1

    if has_baseline:
        base_train = train_df.loc[valid_mask, baseline_col].map(extract_numeric).to_numpy(dtype=float)
        base_test = test_df[baseline_col].map(extract_numeric).to_numpy(dtype=float) \
            if baseline_col in test_df.columns else np.full(len(test_df), np.nan)
    else:
        base_train = np.full(n, np.nan)
        base_test = np.full(len(test_df), np.nan)
    base_train_valid = np.isfinite(base_train) & (base_train > EPS)
    base_test_valid = np.isfinite(base_test) & (base_test > EPS)

    X_test_raw = test_df[feature_cols].copy()

    # ---- cleaning + filtering ----
    numeric_cols, categorical_cols = infer_feature_types(X_train_raw, NUMERIC_LIKE_THRESHOLD)
    X_train = apply_feature_cleaning(X_train_raw, numeric_cols, categorical_cols)
    X_test = apply_feature_cleaning(X_test_raw, numeric_cols, categorical_cols)
    X_train, X_test, drop_summary = drop_sparse_or_constant(X_train, X_test, args.min_feature_non_null_ratio)
    print(f"[INFO] feature filtering: {drop_summary}")
    numeric_cols, categorical_cols = infer_feature_types(X_train, NUMERIC_LIKE_THRESHOLD)

    # ---- prior anchor columns ----
    prior_cols: List[Tuple[str, float]] = []
    if not args.disable_prior_anchor:
        prior_cols = select_prior_columns(
            X_train, z, categorical_cols, args.prior_levels, args.prior_shrinkage, rng_seed
        )
        print(f"[INFO] EB prior hierarchy: {[(c, round(g, 4)) for c, g in prior_cols]}")
    prior_col_names = [c for c, _ in prior_cols]

    # ---- folds (stratified by log-target decile) ----
    decile_edges = np.unique(np.quantile(z, np.linspace(0, 1, 11)))
    strata = np.digitize(z, decile_edges[1:-1])
    try:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=rng_seed)
        folds = list(splitter.split(np.zeros(n), strata))
    except ValueError:
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=rng_seed)
        folds = list(splitter.split(np.zeros(n)))

    # ---- OOF prior (leak-free) + full-train prior for test ----
    prior_oof = np.full(n, np.nan)
    if prior_col_names:
        for tr_idx, va_idx in folds:
            eb = EBPrior(cols=prior_col_names, m=args.prior_shrinkage).fit(X_train.iloc[tr_idx], z[tr_idx])
            prior_oof[va_idx] = eb.predict(X_train.iloc[va_idx])
        eb_full = EBPrior(cols=prior_col_names, m=args.prior_shrinkage).fit(X_train, z)
        prior_test = eb_full.predict(X_test)
    else:
        prior_oof[:] = float(np.median(z))
        prior_test = np.full(len(X_test), float(np.median(z)))

    # anchor in z space: baseline where available, EB prior otherwise
    anchor_oof = np.where(base_train_valid, np.log1p(np.where(base_train_valid, base_train, 0.0)), prior_oof)
    anchor_test = np.where(base_test_valid, np.log1p(np.where(base_test_valid, base_test, 0.0)), prior_test)
    anchor_is_real_baseline = bool(has_baseline and base_train_valid.mean() > 0.5)

    # ---- assemble model features: raw features + anchor-derived features ----
    def add_derived(x_df: pd.DataFrame, prior_z: np.ndarray, base_raw: np.ndarray, base_valid: np.ndarray) -> pd.DataFrame:
        x_df = x_df.copy()
        x_df["__prior_z"] = prior_z
        if has_baseline:
            x_df["__baseline_z"] = np.where(base_valid, np.log1p(np.where(base_valid, base_raw, 0.0)), np.nan)
            x_df["__baseline_minus_prior"] = x_df["__baseline_z"] - x_df["__prior_z"]
        return x_df

    # NOTE: for training folds the prior must be the fold-specific one; we add the
    # OOF prior here (each row's value was computed without its own fold), which is
    # slightly conservative and leak-free for OOF evaluation.
    X_train_m = add_derived(X_train, prior_oof, base_train, base_train_valid)
    X_test_m = add_derived(X_test, prior_test, base_test, base_test_valid)
    numeric_cols_m = numeric_cols + [c for c in X_train_m.columns if c.startswith("__")]
    categorical_cols_m = categorical_cols

    tail_thresh = float(np.quantile(z, args.tail_quantile))
    y_tail = (z >= tail_thresh).astype(int)

    # ---- OOF training ----
    preds_oof: Dict[float, np.ndarray] = {q: np.full(n, np.nan) for q in QUANTILES}
    p_tail_oof = np.zeros(n)
    print(f"[INFO] OOF training: {args.folds} folds x ({len(QUANTILES)} quantile models + tail head) ...")
    for fold_i, (tr_idx, va_idx) in enumerate(folds, start=1):
        pre = build_preprocessor(numeric_cols_m, categorical_cols_m)
        x_tr = pre.fit_transform(X_train_m.iloc[tr_idx])
        x_va = pre.transform(X_train_m.iloc[va_idx])
        for q in QUANTILES:
            reg = make_quantile_regressor(q, args.n_estimators, args.learning_rate, rng_seed)
            reg.fit(x_tr, z[tr_idx])
            preds_oof[q][va_idx] = reg.predict(x_va)
        if len(np.unique(y_tail[tr_idx])) > 1:
            clf = make_classifier(args.n_estimators, args.learning_rate, rng_seed)
            clf.fit(x_tr, y_tail[tr_idx])
            p_tail_oof[va_idx] = clf.predict_proba(x_va)[:, 1]
        print(f"[INFO]   fold {fold_i}/{args.folds} done.")

    # enforce quantile monotonicity
    q_sorted = sorted(QUANTILES)
    stacked = np.sort(np.vstack([preds_oof[q] for q in q_sorted]), axis=0)
    for i, q in enumerate(q_sorted):
        preds_oof[q] = stacked[i]

    # ---- domain weights for tuning ----
    domain_w, domain_auc = None, float("nan")
    if not args.disable_domain_weighting:
        try:
            # raw features only: derived anchor columns (__prior_z etc.) are computed
            # OOF on train but full-fit on test, which a domain classifier would
            # spuriously separate on
            pre_dw = build_preprocessor(numeric_cols, categorical_cols)
            x_all_t = pre_dw.fit_transform(pd.concat([X_train, X_test], axis=0, ignore_index=True))
            domain_w, domain_auc = build_domain_weights(
                x_all_t[:n], x_all_t[n:], args.domain_weight_max, rng_seed
            )
            print(f"[INFO] domain classifier AUC={domain_auc:.3f}; validation reweighting enabled.")
        except Exception as exc:
            print(f"[WARN] domain weighting failed ({exc}); using uniform weights.")

    # ---- tune composition on OOF ----
    choice = tune_composition(
        y_true=y,
        anchor_z=anchor_oof,
        preds_z=preds_oof,
        p_tail=p_tail_oof,
        w=domain_w,
        objective=args.objective,
        median_guard_ratio=args.median_guard_ratio,
        mae_guard_ratio=args.mae_guard_ratio,
        anchor_is_real_baseline=anchor_is_real_baseline,
        min_rel_gain=args.min_rel_gain,
        enable_tail_switch=not args.disable_tail_switch,
    )
    z_pt_oof = compose_point_estimate(anchor_oof, preds_oof, p_tail_oof, choice["lam"], choice["tau"], choice["q_hi"])
    y_pt_oof = np.expm1(z_pt_oof)
    print(f"[INFO] selected lambda={choice['lam']}, tail tau={choice['tau']}, q_hi={choice['q_hi']}, "
          f"fallback_to_anchor={choice['fallback_to_anchor']}")
    print(f"[INFO] OOF model:  {json.dumps({k: round(v, 4) for k, v in choice['metrics'].items()})}")
    print(f"[INFO] OOF anchor: {json.dumps({k: round(v, 4) for k, v in choice['anchor_metrics'].items()})}")

    # ---- conformal calibration (CQR) of the interval ----
    q_lo, q_hi_int = INTERVAL
    conf_scores = np.maximum(preds_oof[q_lo] - z, z - preds_oof[q_hi_int])
    conf_scores = np.maximum(conf_scores, 0.0)
    k = min(len(conf_scores) - 1, int(np.ceil((1 - args.interval_alpha) * (len(conf_scores) + 1))))
    conf_c = float(np.sort(conf_scores)[k])
    cover_raw = float(np.mean((z >= preds_oof[q_lo]) & (z <= preds_oof[q_hi_int])))
    cover_adj = float(np.mean((z >= preds_oof[q_lo] - conf_c) & (z <= preds_oof[q_hi_int] + conf_c)))
    print(f"[INFO] interval coverage: raw={cover_raw:.3f} -> conformal={cover_adj:.3f} (c={conf_c:.4f})")

    width_oof = (preds_oof[q_hi_int] + conf_c) - (preds_oof[q_lo] - conf_c)
    width_sorted = np.sort(width_oof)

    # ---- stratified diagnostic report ----
    strat = stratified_report(y, y_pt_oof, np.expm1(anchor_oof))
    print("[INFO] OOF MAE by target-magnitude bin (anchor vs model, win-rate):")
    for r in strat:
        print(f"        bin{r['bin']} n={r['n']:5d} [{r['target_min']:.3g}, {r['target_max']:.3g}] "
              f"anchor={r['anchor_mae']:.4g} model={r['model_mae']:.4g} win={r['model_win_rate']:.2f}")

    # ---- save OOF diagnostics ----
    oof_df = pd.DataFrame(
        {
            "target": y,
            "anchor": np.expm1(anchor_oof),
            "generated_oof": y_pt_oof,
            "p_tail": p_tail_oof,
            "interval_low": np.expm1(preds_oof[q_lo] - conf_c),
            "interval_high": np.expm1(preds_oof[q_hi_int] + conf_c),
        }
    )
    if args.oof_file:
        oof_df.to_csv(args.oof_file, index=False)
        print(f"[INFO] OOF diagnostics -> {args.oof_file}")

    # ---- full retrain + test inference ----
    print("[INFO] retraining on full data for test inference ...")
    pre_full = build_preprocessor(numeric_cols_m, categorical_cols_m)
    x_full = pre_full.fit_transform(X_train_m)
    x_test_t = pre_full.transform(X_test_m)
    preds_test: Dict[float, np.ndarray] = {}
    for q in QUANTILES:
        reg = make_quantile_regressor(q, args.n_estimators, args.learning_rate, rng_seed)
        reg.fit(x_full, z)
        preds_test[q] = reg.predict(x_test_t)
    stacked_t = np.sort(np.vstack([preds_test[q] for q in q_sorted]), axis=0)
    for i, q in enumerate(q_sorted):
        preds_test[q] = stacked_t[i]
    if len(np.unique(y_tail)) > 1:
        clf_full = make_classifier(args.n_estimators, args.learning_rate, rng_seed)
        clf_full.fit(x_full, y_tail)
        p_tail_test = clf_full.predict_proba(x_test_t)[:, 1]
    else:
        p_tail_test = np.zeros(len(X_test_m))

    z_pt_test = compose_point_estimate(anchor_test, preds_test, p_tail_test, choice["lam"], choice["tau"], choice["q_hi"])
    y_pt_test = np.expm1(z_pt_test)
    lo_test = np.expm1(np.maximum(preds_test[q_lo] - conf_c, 0.0))
    hi_test = np.expm1(preds_test[q_hi_int] + conf_c)
    width_test = (preds_test[q_hi_int] + conf_c) - (preds_test[q_lo] - conf_c)
    trust_test = 1.0 - np.searchsorted(width_sorted, width_test, side="right") / max(len(width_sorted), 1)

    gen_col = args.generated_col
    test_df_out[gen_col] = y_pt_test
    test_df_out[f"{gen_col}_low"] = lo_test
    test_df_out[f"{gen_col}_high"] = hi_test
    test_df_out[f"{gen_col}_trust"] = np.round(trust_test, 4)
    test_df_out[f"{gen_col}_tail_risk"] = np.round(p_tail_test, 4)
    test_df_out.to_csv(args.output_file, index=False)
    print(f"[INFO] predictions -> {args.output_file} "
          f"(columns: {gen_col}, {gen_col}_low/_high, {gen_col}_trust, {gen_col}_tail_risk)")

    # ---- report ----
    report = {
        "mode": "baseline_anchor" if anchor_is_real_baseline else "synthesized_prior_anchor",
        "n_train_valid": n,
        "n_test": int(len(test_df_out)),
        "prior_hierarchy": [{"column": c, "z_mae_gain": g} for c, g in prior_cols],
        "selected": {k: choice[k] for k in ("lam", "tau", "q_hi", "fallback_to_anchor")},
        "objective": args.objective,
        "oof_metrics_model": choice["metrics"],
        "oof_metrics_anchor": choice["anchor_metrics"],
        "oof_stratified": strat,
        "interval": {
            "nominal": 1 - args.interval_alpha,
            "raw_coverage": cover_raw,
            "conformal_coverage": cover_adj,
            "conformal_constant": conf_c,
        },
        "domain_auc": domain_auc,
        "tail_threshold_target": float(np.expm1(tail_thresh)),
        "engineering_summary": eng_summary,
        "feature_filtering": drop_summary,
    }
    Path(args.report_file).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] report -> {args.report_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
