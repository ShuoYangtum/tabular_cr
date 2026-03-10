#!/usr/bin/env python3
"""
Train a robust tree-based regressor from train table and generate predictions for test table.

Goal:
- Stage 1 (classification): predict adaptive target bin and whether baseline under/over-estimates target.
- Stage 2 (regression): use Stage-1 outputs + original features to predict ratio = target / baseline.
- Recover prediction with generated = baseline * ratio_pred.
- Estimate a train ratio interval (default central 95%) and clip predicted ratios into this range.
- Save a new file with prediction column named `generated`.

Supports:
- Mixed numeric/string features
- Missing values
- Noisy numeric strings (e.g. "12.3kg", "about -7", "1,234.5")
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


# =========================
# Editable defaults (quick way)
# =========================
DEFAULT_TRAIN_FILE = "../modified/train_clean.csv"
DEFAULT_TEST_FILE = "../modified/test_clean.csv"
DEFAULT_OUTPUT_FILE = "generated.csv"
DEFAULT_TARGET_COL = "esg-bewertung__input__wasserverbrauch-m3"
DEFAULT_BASELINE_COL = "wasser_berechnet"
DEFAULT_GENERATED_COL = "generated"
DEFAULT_N_ESTIMATORS = 300
DEFAULT_RATIO_COVERAGE = 0.95
DEFAULT_TARGET_BINS = 10
DEFAULT_LLM_MODEL_PATH = "/data/models/Qwen3-4B-Instruct-2507"
DEFAULT_LLM_SAMPLE_SIZE = 20
DEFAULT_LLM_MAX_NEW_TOKENS = 220
DEFAULT_LLM_TEMPERATURE = 0.0

# If an object column has >= this ratio of parseable numeric values, treat it as numeric.
NUMERIC_LIKE_THRESHOLD = 0.85


NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
MISSING_STRINGS = {"", "nan", "none", "null", "na", "n/a", "-", "--", "unknown"}
FLOAT32_SAFE_MAX = np.finfo(np.float32).max * 0.99
BASELINE_EPS = 1e-12


def extract_numeric(value) -> float:
    if value is None:
        return np.nan

    if isinstance(value, (int, float, np.number)):
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return np.nan
        if abs(v) > FLOAT32_SAFE_MAX:
            return np.nan
        return v

    s = str(value).strip()
    if not s:
        return np.nan

    s = s.replace(",", "")
    s = re.sub(r"\s+", "", s)
    if s.lower() in MISSING_STRINGS:
        return np.nan

    match = NUMERIC_PATTERN.search(s)
    if not match:
        return np.nan

    try:
        v = float(match.group())
        if math.isnan(v) or math.isinf(v):
            return np.nan
        if abs(v) > FLOAT32_SAFE_MAX:
            return np.nan
        return v
    except (ValueError, OverflowError):
        return np.nan


def clean_categorical(value) -> str | float:
    if value is None:
        return np.nan
    s = str(value).strip()
    if s.lower() in MISSING_STRINGS:
        return np.nan
    return s


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric features by fitted quantile range to reduce outlier impact."""

    def __init__(self, low_q: float = 0.01, high_q: float = 0.99):
        self.low_q = low_q
        self.high_q = high_q
        self.lower_: np.ndarray | None = None
        self.upper_: np.ndarray | None = None

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(arr, self.low_q, axis=0)
        self.upper_ = np.nanquantile(arr, self.high_q, axis=0)
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.lower_, self.upper_)


def infer_feature_types(df: pd.DataFrame, threshold: float) -> Tuple[List[str], List[str]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
            continue

        non_null = s.notna().sum()
        if non_null == 0:
            categorical_cols.append(col)
            continue

        parsed = s.map(extract_numeric)
        ratio = parsed.notna().sum() / non_null

        if ratio >= threshold:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def apply_feature_cleaning(
    df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]
) -> pd.DataFrame:
    cleaned_cols: Dict[str, pd.Series] = {}
    for col in numeric_cols:
        s = df[col].map(extract_numeric)
        # Replace any residual infinities to avoid sklearn finite-value checks.
        s = s.replace([np.inf, -np.inf], np.nan)
        cleaned_cols[col] = s
    for col in categorical_cols:
        cleaned_cols[col] = df[col].map(clean_categorical)
    # Build once to avoid DataFrame fragmentation from repeated column inserts.
    return pd.DataFrame(cleaned_cols, index=df.index).copy()


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clip", QuantileClipper(low_q=0.01, high_q=0.99)),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
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
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def build_regressor(n_estimators: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        learning_rate=0.05,
        n_estimators=n_estimators,
        max_depth=3,
        min_samples_leaf=20,
        subsample=0.9,
        random_state=42,
        warm_start=True,
    )


def build_classifier(n_estimators: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        learning_rate=0.05,
        n_estimators=n_estimators,
        max_depth=3,
        min_samples_leaf=20,
        subsample=0.9,
        random_state=42,
        warm_start=True,
    )


def make_adaptive_bins(y: np.ndarray, n_bins: int) -> np.ndarray:
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(y, q)
    edges = np.unique(edges)
    if len(edges) < 2:
        center = float(np.mean(y))
        return np.array([center - 1e-6, center + 1e-6], dtype=float)
    return edges


def assign_bins(y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # Return bin index in [0, len(edges)-2]
    return np.digitize(y, edges[1:-1], right=False).astype(int)


def _sample_values_for_llm(series: pd.Series, sample_size: int) -> List[str]:
    s = series.dropna()
    if s.empty:
        return []
    return s.astype(str).drop_duplicates().head(sample_size).tolist()


def _quick_stats_for_llm(series: pd.Series) -> Dict[str, Any]:
    total = int(series.shape[0])
    na_count = int(series.isna().sum())
    non_na = max(total - na_count, 1)
    nunique = int(series.nunique(dropna=True))

    numeric = pd.to_numeric(series, errors="coerce")
    dt = pd.to_datetime(series, errors="coerce", utc=True)

    numeric_ratio = float(numeric.notna().mean()) if total > 0 else 0.0
    datetime_ratio = float(dt.notna().mean()) if total > 0 else 0.0
    unique_ratio = float(nunique / non_na)

    year_like_ratio = 0.0
    if numeric.notna().any():
        n = numeric.dropna()
        year_like_ratio = float(((n >= 1900) & (n <= 2100) & (np.floor(n) == n)).mean())

    return {
        "total": total,
        "na_count": na_count,
        "nunique": nunique,
        "unique_ratio_non_na": round(unique_ratio, 6),
        "numeric_parse_ratio": round(numeric_ratio, 6),
        "datetime_parse_ratio": round(datetime_ratio, 6),
        "year_like_ratio_among_numeric": round(year_like_ratio, 6),
    }


def _safe_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return {}


def infer_feature_semantics_with_llm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    model_path: str,
    sample_size: int,
    max_new_tokens: int,
    temperature: float,
) -> Dict[str, Dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    decisions: Dict[str, Dict[str, Any]] = {}
    total = len(feature_cols)
    for idx, col in enumerate(feature_cols, start=1):
        stats = {
            "train": _quick_stats_for_llm(train_df[col] if col in train_df.columns else pd.Series(dtype="object")),
            "test": _quick_stats_for_llm(test_df[col] if col in test_df.columns else pd.Series(dtype="object")),
        }
        train_samples = _sample_values_for_llm(train_df[col] if col in train_df.columns else pd.Series(dtype="object"), sample_size)
        test_samples = _sample_values_for_llm(test_df[col] if col in test_df.columns else pd.Series(dtype="object"), sample_size)

        schema_hint = {
            "final_type": "one of [id, datetime, year, numeric, categorical, text, unknown]",
            "recommended_actions": {
                "drop_as_feature": "bool",
                "parse_datetime": "bool",
                "extract_datetime_parts": "array like [year,month,day,hour,weekday] or []",
                "treat_as_year": "bool",
            },
        }

        prompt = (
            "You are a tabular feature engineering expert. Return STRICT JSON only.\n"
            f"Schema hint: {json.dumps(schema_hint, ensure_ascii=False)}\n"
            f"Column: {col}\n"
            f"Stats: {json.dumps(stats, ensure_ascii=False)}\n"
            f"Train samples: {json.dumps(train_samples, ensure_ascii=False)}\n"
            f"Test samples: {json.dumps(test_samples, ensure_ascii=False)}\n"
        )
        messages = [
            {"role": "system", "content": "Output valid JSON only."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        parsed = _safe_json_object(generated_text)

        decisions[col] = parsed if parsed else {"final_type": "unknown", "recommended_actions": {}}
        print_progress_bar(idx, total, prefix="LLM feature profiling")

    return decisions


def infer_feature_semantics_heuristic(train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, Any]]:
    decisions: Dict[str, Dict[str, Any]] = {}
    for col in feature_cols:
        s = train_df[col]
        stats = _quick_stats_for_llm(s)
        final_type = "categorical"
        actions = {
            "drop_as_feature": False,
            "parse_datetime": False,
            "extract_datetime_parts": [],
            "treat_as_year": False,
        }
        col_low = col.lower()
        if any(k in col_low for k in ["id", "uuid", "guid", "key"]):
            final_type = "id"
            actions["drop_as_feature"] = True
        elif stats["datetime_parse_ratio"] >= 0.8:
            final_type = "datetime"
            actions["parse_datetime"] = True
            actions["extract_datetime_parts"] = ["year", "month", "day", "hour", "weekday"]
        elif stats["numeric_parse_ratio"] >= 0.8 and stats["year_like_ratio_among_numeric"] >= 0.8:
            final_type = "year"
            actions["treat_as_year"] = True
        elif stats["numeric_parse_ratio"] >= 0.8:
            final_type = "numeric"
        decisions[col] = {"final_type": final_type, "recommended_actions": actions}
    return decisions


def apply_semantic_feature_engineering(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    decisions: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, int]]:
    updated_feature_cols: List[str] = []
    dropped_id_like = 0
    datetime_expanded = 0
    year_transformed = 0

    for col in feature_cols:
        decision = decisions.get(col, {})
        actions = decision.get("recommended_actions", {}) if isinstance(decision, dict) else {}
        final_type = str(decision.get("final_type", "")).lower() if isinstance(decision, dict) else ""

        drop_col = bool(actions.get("drop_as_feature", False)) or final_type == "id"
        parse_datetime = bool(actions.get("parse_datetime", False)) or final_type == "datetime"
        treat_as_year = bool(actions.get("treat_as_year", False)) or final_type == "year"

        if drop_col:
            dropped_id_like += 1
            continue

        if parse_datetime:
            datetime_expanded += 1
            for df in (train_df, test_df):
                dt = pd.to_datetime(df[col], errors="coerce", utc=True)
                df[f"{col}__year"] = dt.dt.year
                df[f"{col}__month"] = dt.dt.month
                df[f"{col}__day"] = dt.dt.day
                df[f"{col}__hour"] = dt.dt.hour
                df[f"{col}__weekday"] = dt.dt.weekday
            updated_feature_cols.extend(
                [f"{col}__year", f"{col}__month", f"{col}__day", f"{col}__hour", f"{col}__weekday"]
            )
            continue

        if treat_as_year:
            year_transformed += 1
            for df in (train_df, test_df):
                year_num = df[col].map(extract_numeric)
                df[f"{col}__year"] = year_num
                df[f"{col}__age_from_2026"] = 2026 - year_num
            updated_feature_cols.extend([f"{col}__year", f"{col}__age_from_2026"])
            continue

        updated_feature_cols.append(col)

    summary = {
        "dropped_id_like_cols": dropped_id_like,
        "datetime_expanded_cols": datetime_expanded,
        "year_transformed_cols": year_transformed,
    }
    return train_df, test_df, updated_feature_cols, summary


def print_progress_bar(current: int, total: int, prefix: str = "Training") -> None:
    width = 30
    ratio = 0.0 if total == 0 else current / total
    done = int(width * ratio)
    bar = "#" * done + "-" * (width - done)
    sys.stdout.write(f"\r{prefix}: [{bar}] {current}/{total}")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def fit_gbdt_with_progress(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    n_estimators: int,
    prefix: str,
) -> Tuple[ColumnTransformer, GradientBoostingRegressor]:
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    x_train_t = preprocessor.fit_transform(x_train)

    model = build_regressor(n_estimators=1)
    for i in range(1, n_estimators + 1):
        model.set_params(n_estimators=i)
        model.fit(x_train_t, y_train)
        print_progress_bar(i, n_estimators, prefix=prefix)

    return preprocessor, model


def fit_gbdt_classifier_with_progress(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    n_estimators: int,
    prefix: str,
) -> Tuple[ColumnTransformer, GradientBoostingClassifier]:
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    x_train_t = preprocessor.fit_transform(x_train)

    model = build_classifier(n_estimators=1)
    for i in range(1, n_estimators + 1):
        model.set_params(n_estimators=i)
        model.fit(x_train_t, y_train)
        print_progress_bar(i, n_estimators, prefix=prefix)

    return preprocessor, model


def fmt(v: float) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "NaN"
    return f"{v:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train robust tree model and generate target predictions.")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE, help="Training CSV path")
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE, help="Test CSV path")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE, help="Output CSV path")
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL, help="Target column name in train file")
    parser.add_argument(
        "--baseline-col",
        default=DEFAULT_BASELINE_COL,
        help=f"Baseline column used for ratio learning (default: {DEFAULT_BASELINE_COL})",
    )
    parser.add_argument(
        "--generated-col",
        default=DEFAULT_GENERATED_COL,
        help="Prediction column name in output file",
    )
    parser.add_argument(
        "--numeric-like-threshold",
        type=float,
        default=NUMERIC_LIKE_THRESHOLD,
        help="Threshold for treating object columns as numeric (0~1).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
        help=f"Number of boosting trees (default: {DEFAULT_N_ESTIMATORS}).",
    )
    parser.add_argument(
        "--target-bins",
        type=int,
        default=DEFAULT_TARGET_BINS,
        help=f"Adaptive target bin count for Stage-1 classification (default: {DEFAULT_TARGET_BINS}).",
    )
    parser.add_argument(
        "--ratio-coverage",
        type=float,
        default=DEFAULT_RATIO_COVERAGE,
        help=(
            "Central coverage for ratio clipping range from train data. "
            "Example: 0.95 -> use [2.5%, 97.5%] ratio quantiles."
        ),
    )
    parser.add_argument(
        "--llm-model-path",
        default=DEFAULT_LLM_MODEL_PATH,
        help=f"Local LLM model path for automatic feature semantics (default: {DEFAULT_LLM_MODEL_PATH})",
    )
    parser.add_argument(
        "--disable-llm-feature-profiler",
        action="store_true",
        help="Disable LLM-based feature semantics and use heuristic rules only.",
    )
    parser.add_argument(
        "--llm-sample-size",
        type=int,
        default=DEFAULT_LLM_SAMPLE_SIZE,
        help=f"Sample values per column for LLM prompt (default: {DEFAULT_LLM_SAMPLE_SIZE}).",
    )
    parser.add_argument(
        "--llm-max-new-tokens",
        type=int,
        default=DEFAULT_LLM_MAX_NEW_TOKENS,
        help=f"LLM max new tokens per column (default: {DEFAULT_LLM_MAX_NEW_TOKENS}).",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=DEFAULT_LLM_TEMPERATURE,
        help=f"LLM temperature (default: {DEFAULT_LLM_TEMPERATURE}).",
    )
    args = parser.parse_args()

    train_path = Path(args.train_file)
    test_path = Path(args.test_file)
    output_path = Path(args.output_file)
    target_col = args.target_col
    baseline_col = args.baseline_col
    generated_col = args.generated_col

    if not train_path.exists():
        print(f"[ERROR] Train file not found: {train_path}")
        return 1
    if not test_path.exists():
        print(f"[ERROR] Test file not found: {test_path}")
        return 1
    if not (0.0 <= args.numeric_like_threshold <= 1.0):
        print("[ERROR] --numeric-like-threshold must be between 0 and 1.")
        return 1
    if args.n_estimators <= 0:
        print("[ERROR] --n-estimators must be a positive integer.")
        return 1
    if args.target_bins < 2:
        print("[ERROR] --target-bins must be >= 2.")
        return 1
    if not (0.0 < args.ratio_coverage <= 1.0):
        print("[ERROR] --ratio-coverage must be in (0, 1].")
        return 1
    if args.llm_sample_size <= 0:
        print("[ERROR] --llm-sample-size must be positive.")
        return 1
    if args.llm_max_new_tokens <= 0:
        print("[ERROR] --llm-max-new-tokens must be positive.")
        return 1

    try:
        train_df = load_csv(train_path)
        test_df = load_csv(test_path)
    except Exception as exc:
        print(f"[ERROR] Failed to read CSV: {exc}")
        return 1

    train_df.columns = [str(c).strip() for c in train_df.columns]
    test_df.columns = [str(c).strip() for c in test_df.columns]

    if target_col not in train_df.columns:
        print(f"[ERROR] target column '{target_col}' not found in train file.")
        print(f"Available columns: {list(train_df.columns)}")
        return 1
    if baseline_col not in train_df.columns:
        print(f"[ERROR] baseline column '{baseline_col}' not found in train file.")
        print(f"Available columns: {list(train_df.columns)}")
        return 1
    if baseline_col not in test_df.columns:
        print(
            f"[ERROR] baseline column '{baseline_col}' not found in test file. "
            "Ratio recovery requires baseline in test."
        )
        print(f"Available columns: {list(test_df.columns)}")
        return 1

    # Build feature set from train columns excluding target and baseline.
    feature_cols = [c for c in train_df.columns if c not in {target_col, baseline_col}]
    if not feature_cols:
        print("[ERROR] No feature columns found after excluding target and baseline columns.")
        return 1

    # Explicitly keep test target untouched and never use it as a feature.
    test_has_target_col = target_col in test_df.columns

    llm_used = False
    llm_mode = "heuristic"
    llm_summary = {
        "dropped_id_like_cols": 0,
        "datetime_expanded_cols": 0,
        "year_transformed_cols": 0,
    }
    feature_semantic_decisions: Dict[str, Dict[str, Any]] = {}
    if not args.disable_llm_feature_profiler:
        try:
            model_path = Path(args.llm_model_path)
            if model_path.exists():
                feature_semantic_decisions = infer_feature_semantics_with_llm(
                    train_df=train_df,
                    test_df=test_df,
                    feature_cols=feature_cols,
                    model_path=str(model_path),
                    sample_size=args.llm_sample_size,
                    max_new_tokens=args.llm_max_new_tokens,
                    temperature=args.llm_temperature,
                )
                llm_used = True
                llm_mode = "llm"
            else:
                print(
                    f"[WARN] LLM model path not found: {model_path}. "
                    "Falling back to heuristic feature semantics."
                )
        except Exception as exc:
            print(f"[WARN] LLM feature profiling failed, fallback to heuristic: {exc}")

    if not feature_semantic_decisions:
        feature_semantic_decisions = infer_feature_semantics_heuristic(train_df, feature_cols)

    train_df, test_df, feature_cols, llm_summary = apply_semantic_feature_engineering(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        decisions=feature_semantic_decisions,
    )
    if not feature_cols:
        print("[ERROR] No feature columns left after semantic feature engineering.")
        return 1

    # Ensure test has all required feature columns (missing ones will be filled with NaN).
    missing_in_test = [c for c in feature_cols if c not in test_df.columns]
    for col in missing_in_test:
        test_df[col] = np.nan

    # Ignore extra columns in test that were not used in train.
    X_train_raw = train_df[feature_cols].copy()
    X_test_raw = test_df[feature_cols].copy()

    target_all = train_df[target_col].map(extract_numeric)
    baseline_train_all = train_df[baseline_col].map(extract_numeric)
    baseline_nonzero_mask = baseline_train_all.abs() > BASELINE_EPS
    ratio_all = target_all / baseline_train_all
    ratio_finite_mask = ratio_all.notna() & np.isfinite(ratio_all) & (ratio_all.abs() <= FLOAT32_SAFE_MAX)
    valid_train_mask = target_all.notna() & baseline_train_all.notna() & baseline_nonzero_mask & ratio_finite_mask
    dropped_target_rows = int(target_all.isna().sum())
    dropped_baseline_rows = int(baseline_train_all.isna().sum())
    dropped_zero_baseline_rows = int((baseline_train_all.notna() & (~baseline_nonzero_mask)).sum())
    dropped_invalid_ratio_rows = int((~ratio_finite_mask).sum())
    dropped_train_rows = int((~valid_train_mask).sum())

    X_train_raw = X_train_raw.loc[valid_train_mask].reset_index(drop=True)
    y_train_target = target_all.loc[valid_train_mask].to_numpy(dtype=float)
    baseline_train = baseline_train_all.loc[valid_train_mask].to_numpy(dtype=float)
    y_train_ratio = ratio_all.loc[valid_train_mask].to_numpy(dtype=float)

    lower_q = (1.0 - args.ratio_coverage) / 2.0
    upper_q = 1.0 - lower_q
    ratio_lower = float(np.quantile(y_train_ratio, lower_q))
    ratio_upper = float(np.quantile(y_train_ratio, upper_q))
    if not np.isfinite(ratio_lower) or not np.isfinite(ratio_upper):
        print("[ERROR] Failed to compute valid ratio quantile range from train data.")
        return 1
    if ratio_lower > ratio_upper:
        ratio_lower, ratio_upper = ratio_upper, ratio_lower

    if len(y_train_ratio) < 20:
        print("[ERROR] Too few valid training rows after cleaning target/baseline (<20).")
        return 1

    numeric_cols, categorical_cols = infer_feature_types(X_train_raw, args.numeric_like_threshold)
    X_train = apply_feature_cleaning(X_train_raw, numeric_cols, categorical_cols)
    X_test = apply_feature_cleaning(X_test_raw, numeric_cols, categorical_cols)

    # Holdout validation using a strict 2-stage pipeline.
    x_tr, x_val, y_ratio_tr, _, y_tar_tr, y_tar_val, base_tr, base_val = train_test_split(
        X_train,
        y_train_ratio,
        y_train_target,
        baseline_train,
        test_size=0.2,
        random_state=42,
    )

    # ----- Stage 1 (validation): target-bin classifier -----
    bin_edges_val = make_adaptive_bins(y_tar_tr, args.target_bins)
    y_bin_tr = assign_bins(y_tar_tr, bin_edges_val)
    target_bin_count_val = len(bin_edges_val) - 1
    stage1_bin_pre_val, stage1_bin_model_val = fit_gbdt_classifier_with_progress(
        x_train=x_tr,
        y_train=y_bin_tr,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        n_estimators=args.n_estimators,
        prefix="Stage1 bin classifier (validation)",
    )
    x_tr_bin_t = stage1_bin_pre_val.transform(x_tr)
    x_val_bin_t = stage1_bin_pre_val.transform(x_val)
    stage1_bin_pred_tr = stage1_bin_model_val.predict(x_tr_bin_t)
    stage1_bin_pred_val = stage1_bin_model_val.predict(x_val_bin_t)
    y_bin_val = assign_bins(y_tar_val, bin_edges_val)
    stage1_bin_acc_val = float(accuracy_score(y_bin_val, stage1_bin_pred_val))

    # ----- Stage 1 (validation): over/under classifier -----
    # 1 -> baseline underestimates target, 0 -> baseline overestimates/equal
    y_under_tr = (base_tr < y_tar_tr).astype(int)
    stage1_ud_pre_val, stage1_ud_model_val = fit_gbdt_classifier_with_progress(
        x_train=x_tr,
        y_train=y_under_tr,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        n_estimators=args.n_estimators,
        prefix="Stage1 under/over classifier (validation)",
    )
    x_tr_ud_t = stage1_ud_pre_val.transform(x_tr)
    x_val_ud_t = stage1_ud_pre_val.transform(x_val)
    stage1_ud_pred_tr = stage1_ud_model_val.predict(x_tr_ud_t)
    stage1_ud_pred_val = stage1_ud_model_val.predict(x_val_ud_t)
    y_under_val = (base_val < y_tar_val).astype(int)
    stage1_under_acc_val = float(accuracy_score(y_under_val, stage1_ud_pred_val))

    # ----- Stage 2 (validation): ratio regression with stage-1 outputs -----
    x_tr_stage2 = x_tr.copy()
    x_val_stage2 = x_val.copy()
    x_tr_stage2["stage1_target_bin_pred"] = stage1_bin_pred_tr.astype(float)
    x_tr_stage2["stage1_under_pred"] = stage1_ud_pred_tr.astype(float)
    x_val_stage2["stage1_target_bin_pred"] = stage1_bin_pred_val.astype(float)
    x_val_stage2["stage1_under_pred"] = stage1_ud_pred_val.astype(float)

    numeric_cols_stage2, categorical_cols_stage2 = infer_feature_types(
        x_tr_stage2, args.numeric_like_threshold
    )

    ratio_lower_val = float(np.quantile(y_ratio_tr, lower_q))
    ratio_upper_val = float(np.quantile(y_ratio_tr, upper_q))
    if ratio_lower_val > ratio_upper_val:
        ratio_lower_val, ratio_upper_val = ratio_upper_val, ratio_lower_val

    stage2_pre_val, stage2_model_val = fit_gbdt_with_progress(
        x_train=x_tr_stage2,
        y_train=y_ratio_tr,
        numeric_cols=numeric_cols_stage2,
        categorical_cols=categorical_cols_stage2,
        n_estimators=args.n_estimators,
        prefix="Stage2 ratio regressor (validation)",
    )
    x_val_stage2_t = stage2_pre_val.transform(x_val_stage2)
    val_ratio_pred = stage2_model_val.predict(x_val_stage2_t)
    val_ratio_pred = np.clip(val_ratio_pred, ratio_lower_val, ratio_upper_val)
    val_pred = base_val * val_ratio_pred

    val_mae = mean_absolute_error(y_tar_val, val_pred)
    val_rmse = math.sqrt(mean_squared_error(y_tar_val, val_pred))
    val_r2 = r2_score(y_tar_val, val_pred) if len(y_tar_val) >= 2 else np.nan

    baseline_val_mae = mean_absolute_error(y_tar_val, base_val)
    baseline_val_rmse = math.sqrt(mean_squared_error(y_tar_val, base_val))
    baseline_val_r2 = r2_score(y_tar_val, base_val) if len(y_tar_val) >= 2 else np.nan

    # ----- Final full-train pipeline -----
    bin_edges_final = make_adaptive_bins(y_train_target, args.target_bins)
    y_bin_full = assign_bins(y_train_target, bin_edges_final)
    target_bin_count_final = len(bin_edges_final) - 1
    y_under_full = (baseline_train < y_train_target).astype(int)

    stage1_bin_pre_final, stage1_bin_model_final = fit_gbdt_classifier_with_progress(
        x_train=X_train,
        y_train=y_bin_full,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        n_estimators=args.n_estimators,
        prefix="Stage1 bin classifier (final)",
    )
    stage1_ud_pre_final, stage1_ud_model_final = fit_gbdt_classifier_with_progress(
        x_train=X_train,
        y_train=y_under_full,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        n_estimators=args.n_estimators,
        prefix="Stage1 under/over classifier (final)",
    )

    x_train_bin_t = stage1_bin_pre_final.transform(X_train)
    x_test_bin_t = stage1_bin_pre_final.transform(X_test)
    stage1_bin_pred_train = stage1_bin_model_final.predict(x_train_bin_t)
    stage1_bin_pred_test = stage1_bin_model_final.predict(x_test_bin_t)

    x_train_ud_t = stage1_ud_pre_final.transform(X_train)
    x_test_ud_t = stage1_ud_pre_final.transform(X_test)
    stage1_ud_pred_train = stage1_ud_model_final.predict(x_train_ud_t)
    stage1_ud_pred_test = stage1_ud_model_final.predict(x_test_ud_t)

    x_train_stage2 = X_train.copy()
    x_test_stage2 = X_test.copy()
    x_train_stage2["stage1_target_bin_pred"] = stage1_bin_pred_train.astype(float)
    x_train_stage2["stage1_under_pred"] = stage1_ud_pred_train.astype(float)
    x_test_stage2["stage1_target_bin_pred"] = stage1_bin_pred_test.astype(float)
    x_test_stage2["stage1_under_pred"] = stage1_ud_pred_test.astype(float)

    numeric_cols_stage2_final, categorical_cols_stage2_final = infer_feature_types(
        x_train_stage2, args.numeric_like_threshold
    )

    stage2_pre_final, stage2_model_final = fit_gbdt_with_progress(
        x_train=x_train_stage2,
        y_train=y_train_ratio,
        numeric_cols=numeric_cols_stage2_final,
        categorical_cols=categorical_cols_stage2_final,
        n_estimators=args.n_estimators,
        prefix="Stage2 ratio regressor (final)",
    )
    x_test_stage2_t = stage2_pre_final.transform(x_test_stage2)
    test_ratio_pred = stage2_model_final.predict(x_test_stage2_t)
    test_ratio_pred = np.clip(test_ratio_pred, ratio_lower, ratio_upper)
    baseline_test = test_df[baseline_col].map(extract_numeric).to_numpy(dtype=float)
    test_pred = baseline_test * test_ratio_pred
    missing_test_baseline_rows = int(np.isnan(baseline_test).sum())

    output_df = test_df.copy()
    output_df[generated_col] = test_pred

    try:
        output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    except Exception as exc:
        print(f"[ERROR] Failed to write output: {exc}")
        return 1

    print("=== Training Summary ===")
    print(f"Train file: {train_path}")
    print(f"Test file: {test_path}")
    print(f"Output file: {output_path}")
    print(f"Target column: {target_col}")
    print(f"Baseline column: {baseline_col}")
    print(f"Generated column: {generated_col}")
    print(f"Feature semantics mode: {llm_mode}")
    print(f"LLM profiling used: {'yes' if llm_used else 'no'}")
    print(f"Dropped id-like columns: {llm_summary['dropped_id_like_cols']}")
    print(f"Datetime-expanded columns: {llm_summary['datetime_expanded_cols']}")
    print(f"Year-transformed columns: {llm_summary['year_transformed_cols']}")
    print(f"Stage1 target bins requested: {args.target_bins}")
    print(f"Stage1 target bins used (validation/final): {target_bin_count_val}/{target_bin_count_final}")
    print(f"Ratio coverage: {args.ratio_coverage:.2%}")
    print(f"Ratio clip range (validation): [{fmt(ratio_lower_val)}, {fmt(ratio_upper_val)}]")
    print(f"Ratio clip range (final): [{fmt(ratio_lower)}, {fmt(ratio_upper)}]")
    print(f"Original train rows: {len(train_df)}")
    print(f"Valid train rows used: {len(y_train_ratio)}")
    print(f"Dropped rows (invalid target): {dropped_target_rows}")
    print(f"Dropped rows (invalid baseline): {dropped_baseline_rows}")
    print(f"Dropped rows (baseline approximately zero): {dropped_zero_baseline_rows}")
    print(f"Dropped rows (invalid target/baseline ratio): {dropped_invalid_ratio_rows}")
    print(f"Dropped rows (invalid target/baseline union): {dropped_train_rows}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    if test_has_target_col:
        print("Test file contains target column: yes (excluded from features)")
    else:
        print("Test file contains target column: no")
    if missing_in_test:
        print(f"Missing train features auto-filled in test: {len(missing_in_test)}")
    if missing_test_baseline_rows > 0:
        print(
            "Rows with missing/invalid baseline in test: "
            f"{missing_test_baseline_rows} (generated becomes NaN for these rows)"
        )
    print()

    print("=== Holdout Metrics (20% validation split) ===")
    print(f"Stage1 bin accuracy : {stage1_bin_acc_val:.6f}")
    print(f"Stage1 under/over accuracy : {stage1_under_acc_val:.6f}")
    print("---")
    print(f"Baseline MAE  : {fmt(baseline_val_mae)}")
    print(f"Baseline RMSE : {fmt(baseline_val_rmse)}")
    print(f"Baseline R2   : {fmt(baseline_val_r2)}")
    print("---")
    print(f"MAE  : {fmt(val_mae)}")
    print(f"RMSE : {fmt(val_rmse)}")
    print(f"R2   : {fmt(val_r2)}")
    print()
    print("Prediction complete. New file written with generated column.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
