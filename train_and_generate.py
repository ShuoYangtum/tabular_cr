#!/usr/bin/env python3
"""
Train a robust tree-based regressor from train table and generate predictions for test table.

Goal:
- Train a conservative correction model over baseline:
  generated = baseline + alpha * gate_prob * correction
- correction is learned in signed-log space for stability.
- alpha is chosen on validation from a candidate grid.
- Keep generated/baseline ratio in a train-derived interval (default central 95%).
- Save a new file with prediction column named `generated`.

Supports:
- Mixed numeric/string features
- Missing values
- Noisy numeric strings (e.g. "12.3kg", "about -7", "1,234.5")
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


# =========================
# Editable defaults (quick way)
# =========================
DEFAULT_TRAIN_FILE = "../modified/train_3_clean.csv"
DEFAULT_TEST_FILE = "../modified/test_3_clean.csv"
DEFAULT_OUTPUT_FILE = "generated.csv"
DEFAULT_TARGET_COL = "esg_firma_esg-bewertung__input__wasserverbrauch-m3"
DEFAULT_BASELINE_COL = "esg_firma_wasser_berechnet"
DEFAULT_GENERATED_COL = "generated"
DEFAULT_N_ESTIMATORS = 300
DEFAULT_RATIO_COVERAGE = 0.95
DEFAULT_LLM_MODEL_PATH = "/data/models/Qwen3-4B-Instruct-2507"
DEFAULT_LLM_SAMPLE_SIZE = 20
DEFAULT_LLM_MAX_NEW_TOKENS = 220
DEFAULT_LLM_TEMPERATURE = 0.0
DEFAULT_FEATURE_SEMANTIC_CACHE = "feature_semantic_cache.json"
DEFAULT_MODEL_TYPE = "tabpfn"
DEFAULT_ALPHA_GRID = "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
DEFAULT_GATE_QUANTILE = 0.6
DEFAULT_MIN_FEATURE_NON_NULL_RATIO = 0.01
DEFAULT_BASELINE_ALPHA_BINS = 10
DEFAULT_MIN_BIN_ROWS = 80
DEFAULT_GATE_QUANTILE_CANDIDATES = "0.5,0.6,0.7,0.8"
DEFAULT_TUNE_OBJECTIVE = "hybrid"
DEFAULT_MIN_VALIDATION_REL_GAIN = 0.002

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


def build_regressor(model_type: str, n_estimators: int):
    if model_type == "hgbt":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=n_estimators,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1e-3,
            random_state=42,
            warm_start=True,
        )
    return GradientBoostingRegressor(
        learning_rate=0.05,
        n_estimators=n_estimators,
        max_depth=3,
        min_samples_leaf=20,
        subsample=0.9,
        random_state=42,
        warm_start=True,
    )


def build_classifier(model_type: str, n_estimators: int):
    if model_type == "hgbt":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=n_estimators,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1e-3,
            random_state=42,
            warm_start=True,
        )
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
    dt = parse_datetime_series(series)

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


def parse_datetime_series(series: pd.Series) -> pd.Series:
    # Prefer mixed format parsing (pandas>=2) and silence format-inference warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not infer format.*", category=UserWarning)
        try:
            return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
        except TypeError:
            return pd.to_datetime(series, errors="coerce", utc=True)


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
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
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
        inputs = tokenizer([text], return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
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


def build_semantic_cache_fingerprint(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> str:
    payload = {
        "train_columns": list(train_df.columns),
        "test_columns": list(test_df.columns),
        "feature_cols": list(feature_cols),
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_semantic_cache(cache_path: Path, fingerprint: str) -> Dict[str, Dict[str, Any]] | None:
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("fingerprint") != fingerprint:
            return None
        decisions = data.get("decisions")
        if isinstance(decisions, dict):
            return decisions
        return None
    except Exception:
        return None


def save_semantic_cache(cache_path: Path, fingerprint: str, decisions: Dict[str, Dict[str, Any]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "fingerprint": fingerprint,
        "decisions": decisions,
    }
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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
    train_new_cols: Dict[str, pd.Series] = {}
    test_new_cols: Dict[str, pd.Series] = {}

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
            dt_train = parse_datetime_series(train_df[col])
            dt_test = parse_datetime_series(test_df[col])
            train_new_cols[f"{col}__year"] = dt_train.dt.year
            train_new_cols[f"{col}__month"] = dt_train.dt.month
            train_new_cols[f"{col}__day"] = dt_train.dt.day
            train_new_cols[f"{col}__hour"] = dt_train.dt.hour
            train_new_cols[f"{col}__weekday"] = dt_train.dt.weekday
            test_new_cols[f"{col}__year"] = dt_test.dt.year
            test_new_cols[f"{col}__month"] = dt_test.dt.month
            test_new_cols[f"{col}__day"] = dt_test.dt.day
            test_new_cols[f"{col}__hour"] = dt_test.dt.hour
            test_new_cols[f"{col}__weekday"] = dt_test.dt.weekday
            updated_feature_cols.extend(
                [f"{col}__year", f"{col}__month", f"{col}__day", f"{col}__hour", f"{col}__weekday"]
            )
            continue

        if treat_as_year:
            year_transformed += 1
            train_year = train_df[col].map(extract_numeric)
            test_year = test_df[col].map(extract_numeric)
            train_new_cols[f"{col}__year"] = train_year
            train_new_cols[f"{col}__age_from_2026"] = 2026 - train_year
            test_new_cols[f"{col}__year"] = test_year
            test_new_cols[f"{col}__age_from_2026"] = 2026 - test_year
            updated_feature_cols.extend([f"{col}__year", f"{col}__age_from_2026"])
            continue

        updated_feature_cols.append(col)

    if train_new_cols:
        train_df = pd.concat([train_df, pd.DataFrame(train_new_cols, index=train_df.index)], axis=1).copy()
    else:
        train_df = train_df.copy()
    if test_new_cols:
        test_df = pd.concat([test_df, pd.DataFrame(test_new_cols, index=test_df.index)], axis=1).copy()
    else:
        test_df = test_df.copy()

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
    model_type: str,
):
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    x_train_t = preprocessor.fit_transform(x_train)

    if model_type == "tabpfn":
        try:
            from tabpfn import TabPFNRegressor
        except Exception as exc:
            raise RuntimeError(
                "model_type='tabpfn' requires the 'tabpfn' package. Install with: pip install tabpfn"
            ) from exc
        model = TabPFNRegressor()
        print_progress_bar(0, 1, prefix=prefix)
        model.fit(x_train_t, y_train)
        print_progress_bar(1, 1, prefix=prefix)
        return preprocessor, model

    model = build_regressor(model_type=model_type, n_estimators=1)
    for i in range(1, n_estimators + 1):
        if model_type == "hgbt":
            model.set_params(max_iter=i)
        else:
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
    model_type: str,
):
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    x_train_t = preprocessor.fit_transform(x_train)

    if model_type == "tabpfn":
        try:
            from tabpfn import TabPFNClassifier
        except Exception as exc:
            raise RuntimeError(
                "model_type='tabpfn' requires the 'tabpfn' package. Install with: pip install tabpfn"
            ) from exc
        model = TabPFNClassifier()
        print_progress_bar(0, 1, prefix=prefix)
        model.fit(x_train_t, y_train)
        print_progress_bar(1, 1, prefix=prefix)
        return preprocessor, model

    model = build_classifier(model_type=model_type, n_estimators=1)
    for i in range(1, n_estimators + 1):
        if model_type == "hgbt":
            model.set_params(max_iter=i)
        else:
            model.set_params(n_estimators=i)
        model.fit(x_train_t, y_train)
        print_progress_bar(i, n_estimators, prefix=prefix)

    return preprocessor, model


def fmt(v: float) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "NaN"
    return f"{v:.6f}"


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def signed_expm1(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.expm1(np.abs(x))


def parse_alpha_grid(grid_text: str) -> List[float]:
    parts = [p.strip() for p in str(grid_text).split(",") if p.strip()]
    if not parts:
        raise ValueError("alpha grid is empty")
    out: List[float] = []
    for p in parts:
        v = float(p)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"alpha must be in [0,1], got {v}")
        out.append(v)
    out = sorted(set(out))
    return out


def parse_float_list(text: str) -> List[float]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        raise ValueError("float list is empty")
    return [float(p) for p in parts]


def trimmed_mean(arr: np.ndarray, trim_ratio: float) -> float:
    if len(arr) == 0:
        return np.nan
    if not (0 <= trim_ratio < 0.5):
        raise ValueError("trim_ratio must be in [0, 0.5).")
    lo = int(len(arr) * trim_ratio)
    hi = len(arr) - lo
    if hi <= lo:
        return np.nan
    arr_sorted = np.sort(arr)
    return float(np.mean(arr_sorted[lo:hi]))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan
    trimmed_rmse90 = float(np.sqrt(trimmed_mean(err**2, 0.05)))
    return {"mae": mae, "rmse": rmse, "r2": r2, "trimmed_rmse90": trimmed_rmse90}


def objective_value(metrics: Dict[str, float], objective: str) -> float:
    if objective == "rmse":
        return float(metrics["rmse"])
    if objective == "trimmed_rmse90":
        return float(metrics["trimmed_rmse90"])
    # hybrid objective
    return float(0.6 * metrics["rmse"] + 0.4 * metrics["trimmed_rmse90"])


def drop_high_unique_id_like_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    unique_ratio_threshold: float = 0.98,
    min_non_null_rows: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    id_hints = ("id", "uuid", "guid", "key", "nummer", "crefonummer", "timestamp", "zeitpunkt")
    keep_cols: List[str] = []
    dropped_id_like = 0

    for col in x_train.columns:
        s = x_train[col]
        non_null = int(s.notna().sum())
        if non_null < min_non_null_rows:
            keep_cols.append(col)
            continue
        uniq_ratio = float(s.nunique(dropna=True) / max(non_null, 1))
        col_low = col.lower()
        has_hint = any(h in col_low for h in id_hints)
        if uniq_ratio >= unique_ratio_threshold and has_hint:
            dropped_id_like += 1
            continue
        keep_cols.append(col)

    return (
        x_train[keep_cols].copy(),
        x_test[keep_cols].copy(),
        {"dropped_high_unique_id_like_cols": dropped_id_like},
    )


def pick_best_alpha(
    y_true: np.ndarray,
    base: np.ndarray,
    corr_pred: np.ndarray,
    gate_prob: np.ndarray,
    alpha_candidates: List[float],
    ratio_lower: float,
    ratio_upper: float,
    objective: str,
) -> Tuple[float, Dict[str, float]]:
    base_slog = signed_log1p(base)
    best_alpha = alpha_candidates[0]
    best_score = float("inf")
    best_metrics = {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "trimmed_rmse90": np.nan}
    for alpha in alpha_candidates:
        y_slog_pred = base_slog + alpha * gate_prob * corr_pred
        y_pred = signed_expm1(y_slog_pred)
        y_ratio_pred = y_pred / base
        y_ratio_pred = np.clip(y_ratio_pred, ratio_lower, ratio_upper)
        y_pred = base * y_ratio_pred
        metrics = evaluate_predictions(y_true, y_pred)
        score = objective_value(metrics, objective)
        if score < best_score:
            best_score = score
            best_alpha = alpha
            best_metrics = metrics
    return best_alpha, best_metrics


def fit_segmented_alpha(
    y_true: np.ndarray,
    base: np.ndarray,
    corr_pred: np.ndarray,
    gate_prob: np.ndarray,
    alpha_candidates: List[float],
    ratio_lower: float,
    ratio_upper: float,
    n_bins: int,
    min_bin_rows: int,
    objective: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    edges = make_adaptive_bins(base, n_bins)
    bin_idx = assign_bins(base, edges)
    default_alpha, _ = pick_best_alpha(
        y_true=y_true,
        base=base,
        corr_pred=corr_pred,
        gate_prob=gate_prob,
        alpha_candidates=alpha_candidates,
        ratio_lower=ratio_lower,
        ratio_upper=ratio_upper,
        objective=objective,
    )

    alpha_by_bin = np.full(len(edges) - 1, default_alpha, dtype=float)
    for b in range(len(alpha_by_bin)):
        mask = bin_idx == b
        if int(mask.sum()) < min_bin_rows:
            continue
        best_a, _ = pick_best_alpha(
            y_true=y_true[mask],
            base=base[mask],
            corr_pred=corr_pred[mask],
            gate_prob=gate_prob[mask],
            alpha_candidates=alpha_candidates,
            ratio_lower=ratio_lower,
            ratio_upper=ratio_upper,
            objective=objective,
        )
        alpha_by_bin[b] = best_a

    pred_slog = signed_log1p(base) + alpha_by_bin[bin_idx] * gate_prob * corr_pred
    pred = signed_expm1(pred_slog)
    pred_ratio = pred / base
    pred_ratio = np.clip(pred_ratio, ratio_lower, ratio_upper)
    pred = base * pred_ratio
    metrics = evaluate_predictions(y_true, pred)
    return edges, alpha_by_bin, metrics


def drop_sparse_or_constant_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    min_non_null_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    keep_cols: List[str] = []
    dropped_sparse = 0
    dropped_constant = 0
    for col in x_train.columns:
        non_null_ratio = float(x_train[col].notna().mean())
        if non_null_ratio < min_non_null_ratio:
            dropped_sparse += 1
            continue
        # drop columns with no variation in train after cleaning
        nunique = int(x_train[col].nunique(dropna=True))
        if nunique <= 1:
            dropped_constant += 1
            continue
        keep_cols.append(col)
    return (
        x_train[keep_cols].copy(),
        x_test[keep_cols].copy(),
        {
            "kept_feature_cols": len(keep_cols),
            "dropped_sparse_cols": dropped_sparse,
            "dropped_constant_cols": dropped_constant,
        },
    )


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
        "--model-type",
        default=DEFAULT_MODEL_TYPE,
        choices=["gbdt", "hgbt", "tabpfn"],
        help=f"Model family for regressor/classifier (default: {DEFAULT_MODEL_TYPE}).",
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
        "--feature-semantic-cache",
        default=DEFAULT_FEATURE_SEMANTIC_CACHE,
        help=f"Cache file for LLM feature semantics (default: {DEFAULT_FEATURE_SEMANTIC_CACHE}).",
    )
    parser.add_argument(
        "--refresh-feature-semantic-cache",
        action="store_true",
        help="Ignore existing feature semantic cache and rerun LLM profiling.",
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
    parser.add_argument(
        "--alpha-grid",
        default=DEFAULT_ALPHA_GRID,
        help=f"Comma-separated alpha candidates for blending (default: {DEFAULT_ALPHA_GRID})",
    )
    parser.add_argument(
        "--gate-quantile",
        type=float,
        default=DEFAULT_GATE_QUANTILE,
        help=f"Quantile for gate label on |log-correction| (default: {DEFAULT_GATE_QUANTILE}).",
    )
    parser.add_argument(
        "--gate-quantile-candidates",
        default=DEFAULT_GATE_QUANTILE_CANDIDATES,
        help=(
            "Comma-separated candidates for auto gate-quantile search on validation "
            f"(default: {DEFAULT_GATE_QUANTILE_CANDIDATES})."
        ),
    )
    parser.add_argument(
        "--disable-auto-gate-quantile",
        action="store_true",
        help="Disable automatic gate-quantile search and use --gate-quantile only.",
    )
    parser.add_argument(
        "--baseline-alpha-bins",
        type=int,
        default=DEFAULT_BASELINE_ALPHA_BINS,
        help=f"Number of baseline bins for segmented alpha (default: {DEFAULT_BASELINE_ALPHA_BINS}).",
    )
    parser.add_argument(
        "--min-bin-rows",
        type=int,
        default=DEFAULT_MIN_BIN_ROWS,
        help=f"Minimum validation rows to tune alpha per bin (default: {DEFAULT_MIN_BIN_ROWS}).",
    )
    parser.add_argument(
        "--tune-objective",
        default=DEFAULT_TUNE_OBJECTIVE,
        choices=["rmse", "trimmed_rmse90", "hybrid"],
        help=f"Validation tuning objective (default: {DEFAULT_TUNE_OBJECTIVE}).",
    )
    parser.add_argument(
        "--min-validation-rel-gain",
        type=float,
        default=DEFAULT_MIN_VALIDATION_REL_GAIN,
        help=(
            "Minimum relative objective improvement over baseline required to apply correction "
            f"(default: {DEFAULT_MIN_VALIDATION_REL_GAIN})."
        ),
    )
    parser.add_argument(
        "--min-feature-non-null-ratio",
        type=float,
        default=DEFAULT_MIN_FEATURE_NON_NULL_RATIO,
        help=(
            "Drop feature columns with train non-null ratio below this threshold "
            f"(default: {DEFAULT_MIN_FEATURE_NON_NULL_RATIO})."
        ),
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
    if not (0.0 < args.ratio_coverage <= 1.0):
        print("[ERROR] --ratio-coverage must be in (0, 1].")
        return 1
    if not (0.0 < args.gate_quantile < 1.0):
        print("[ERROR] --gate-quantile must be in (0, 1).")
        return 1
    if args.baseline_alpha_bins < 2:
        print("[ERROR] --baseline-alpha-bins must be >= 2.")
        return 1
    if args.min_bin_rows <= 0:
        print("[ERROR] --min-bin-rows must be positive.")
        return 1
    if args.min_validation_rel_gain < 0:
        print("[ERROR] --min-validation-rel-gain must be >= 0.")
        return 1
    if not (0.0 <= args.min_feature_non_null_ratio < 1.0):
        print("[ERROR] --min-feature-non-null-ratio must be in [0, 1).")
        return 1
    if args.llm_sample_size <= 0:
        print("[ERROR] --llm-sample-size must be positive.")
        return 1
    if args.llm_max_new_tokens <= 0:
        print("[ERROR] --llm-max-new-tokens must be positive.")
        return 1
    try:
        alpha_candidates = parse_alpha_grid(args.alpha_grid)
    except Exception as exc:
        print(f"[ERROR] Invalid --alpha-grid: {exc}")
        return 1
    try:
        gate_quantile_candidates = parse_float_list(args.gate_quantile_candidates)
    except Exception as exc:
        print(f"[ERROR] Invalid --gate-quantile-candidates: {exc}")
        return 1
    gate_quantile_candidates = [q for q in gate_quantile_candidates if 0.0 < q < 1.0]
    if not gate_quantile_candidates:
        print("[ERROR] No valid gate quantile candidates in (0,1).")
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
    llm_cache_used = False
    llm_summary = {
        "dropped_id_like_cols": 0,
        "datetime_expanded_cols": 0,
        "year_transformed_cols": 0,
    }
    feature_semantic_decisions: Dict[str, Dict[str, Any]] = {}
    semantic_cache_path = Path(args.feature_semantic_cache)
    semantic_fingerprint = build_semantic_cache_fingerprint(train_df, test_df, feature_cols)

    if not args.refresh_feature_semantic_cache:
        cached_decisions = load_semantic_cache(semantic_cache_path, semantic_fingerprint)
        if cached_decisions:
            feature_semantic_decisions = cached_decisions
            llm_mode = "cache"
            llm_cache_used = True

    if (not feature_semantic_decisions) and (not args.disable_llm_feature_profiler):
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
                save_semantic_cache(semantic_cache_path, semantic_fingerprint, feature_semantic_decisions)
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

    X_train, X_test, feature_drop_summary = drop_sparse_or_constant_features(
        X_train,
        X_test,
        min_non_null_ratio=args.min_feature_non_null_ratio,
    )
    X_train, X_test, id_like_drop_summary = drop_high_unique_id_like_features(
        X_train,
        X_test,
        unique_ratio_threshold=0.98,
        min_non_null_rows=200,
    )
    feature_drop_summary.update(id_like_drop_summary)
    if X_train.shape[1] == 0:
        print("[ERROR] No usable feature columns left after sparse/constant filtering.")
        return 1
    feature_cols = list(X_train.columns)
    numeric_cols, categorical_cols = infer_feature_types(X_train, args.numeric_like_threshold)

    # Holdout validation using conservative correction framework.
    x_tr, x_val, y_ratio_tr, _, y_tar_tr, y_tar_val, base_tr, base_val = train_test_split(
        X_train,
        y_train_ratio,
        y_train_target,
        baseline_train,
        test_size=0.2,
        random_state=42,
    )
    ratio_lower_val = float(np.quantile(y_ratio_tr, lower_q))
    ratio_upper_val = float(np.quantile(y_ratio_tr, upper_q))
    if ratio_lower_val > ratio_upper_val:
        ratio_lower_val, ratio_upper_val = ratio_upper_val, ratio_lower_val

    # Regressor target: correction in signed-log space
    y_corr_tr = signed_log1p(y_tar_tr) - signed_log1p(base_tr)
    y_corr_val_true = signed_log1p(y_tar_val) - signed_log1p(base_val)
    corr_pre_val, corr_model_val = fit_gbdt_with_progress(
        x_train=x_tr,
        y_train=y_corr_tr,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        n_estimators=args.n_estimators,
        prefix="Correction regressor (validation)",
        model_type=args.model_type,
    )
    x_val_corr_t = corr_pre_val.transform(x_val)
    corr_pred_val = corr_model_val.predict(x_val_corr_t)

    # Gate classifier with optional automatic quantile tuning.
    baseline_val_mae = mean_absolute_error(y_tar_val, base_val)
    baseline_val_rmse = math.sqrt(mean_squared_error(y_tar_val, base_val))
    baseline_val_r2 = r2_score(y_tar_val, base_val) if len(y_tar_val) >= 2 else np.nan
    baseline_val_metrics = evaluate_predictions(y_tar_val, base_val)

    if args.disable_auto_gate_quantile:
        q_candidates = [args.gate_quantile]
    else:
        q_candidates = sorted(set(gate_quantile_candidates + [args.gate_quantile]))

    best_val_package: Dict[str, Any] = {}
    best_val_rmse = float("inf")

    for q in q_candidates:
        gate_thr_cand = float(np.quantile(np.abs(y_corr_tr), q))
        y_gate_tr = (np.abs(y_corr_tr) >= gate_thr_cand).astype(int)
        gate_pre_cand, gate_model_cand = fit_gbdt_classifier_with_progress(
            x_train=x_tr,
            y_train=y_gate_tr,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            n_estimators=args.n_estimators,
            prefix=f"Correction gate classifier (validation, q={q:.2f})",
            model_type=args.model_type,
        )
        x_val_gate_t = gate_pre_cand.transform(x_val)
        gate_pred_val = gate_model_cand.predict(x_val_gate_t)
        gate_acc_cand = float(
            accuracy_score((np.abs(y_corr_val_true) >= gate_thr_cand).astype(int), gate_pred_val)
        )
        if hasattr(gate_model_cand, "predict_proba"):
            gate_prob_val = gate_model_cand.predict_proba(x_val_gate_t)[:, 1]
        else:
            gate_prob_val = gate_pred_val.astype(float)

        # Global alpha
        alpha_global, metrics_global = pick_best_alpha(
            y_true=y_tar_val,
            base=base_val,
            corr_pred=corr_pred_val,
            gate_prob=gate_prob_val,
            alpha_candidates=alpha_candidates,
            ratio_lower=ratio_lower_val,
            ratio_upper=ratio_upper_val,
            objective=args.tune_objective,
        )

        # Segmented alpha by baseline bins
        alpha_edges_cand, alpha_by_bin_cand, metrics_segment = fit_segmented_alpha(
            y_true=y_tar_val,
            base=base_val,
            corr_pred=corr_pred_val,
            gate_prob=gate_prob_val,
            alpha_candidates=alpha_candidates,
            ratio_lower=ratio_lower_val,
            ratio_upper=ratio_upper_val,
            n_bins=args.baseline_alpha_bins,
            min_bin_rows=args.min_bin_rows,
            objective=args.tune_objective,
        )

        use_segment = objective_value(metrics_segment, args.tune_objective) < objective_value(
            metrics_global, args.tune_objective
        )
        selected_metrics = metrics_segment if use_segment else metrics_global
        selected_score = objective_value(selected_metrics, args.tune_objective)

        if selected_score < best_val_rmse:
            best_val_rmse = selected_score
            best_val_package = {
                "q": q,
                "gate_thr": gate_thr_cand,
                "gate_acc": gate_acc_cand,
                "gate_pre": gate_pre_cand,
                "gate_model": gate_model_cand,
                "alpha_global": alpha_global,
                "alpha_edges": alpha_edges_cand,
                "alpha_by_bin": alpha_by_bin_cand,
                "use_segment_alpha": use_segment,
                "metrics_global": metrics_global,
                "metrics_segment": metrics_segment,
                "metrics_selected": selected_metrics,
                "selected_score": selected_score,
            }

    selected_gate_quantile = float(best_val_package["q"])
    gate_thr = float(best_val_package["gate_thr"])
    gate_acc_val = float(best_val_package["gate_acc"])
    best_alpha = float(best_val_package["alpha_global"])
    best_alpha_edges = np.asarray(best_val_package["alpha_edges"], dtype=float)
    best_alpha_by_bin = np.asarray(best_val_package["alpha_by_bin"], dtype=float)
    use_segment_alpha = bool(best_val_package["use_segment_alpha"])
    val_mae = float(best_val_package["metrics_selected"]["mae"])
    val_rmse = float(best_val_package["metrics_selected"]["rmse"])
    val_r2 = float(best_val_package["metrics_selected"]["r2"])
    val_trimmed_rmse90 = float(best_val_package["metrics_selected"]["trimmed_rmse90"])

    baseline_objective = objective_value(baseline_val_metrics, args.tune_objective)
    selected_objective = float(best_val_package["selected_score"])
    rel_gain = (baseline_objective - selected_objective) / max(abs(baseline_objective), 1e-12)
    fallback_to_baseline = rel_gain < args.min_validation_rel_gain
    if fallback_to_baseline:
        best_alpha = 0.0
        use_segment_alpha = False
        val_mae = float(baseline_val_metrics["mae"])
        val_rmse = float(baseline_val_metrics["rmse"])
        val_r2 = float(baseline_val_metrics["r2"])
        val_trimmed_rmse90 = float(baseline_val_metrics["trimmed_rmse90"])

    # ----- Final full-train models -----
    y_corr_full = signed_log1p(y_train_target) - signed_log1p(baseline_train)
    corr_pre_final, corr_model_final = fit_gbdt_with_progress(
        x_train=X_train,
        y_train=y_corr_full,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        n_estimators=args.n_estimators,
        prefix="Correction regressor (final)",
        model_type=args.model_type,
    )
    gate_thr_full = float(np.quantile(np.abs(y_corr_full), selected_gate_quantile))
    y_gate_full = (np.abs(y_corr_full) >= gate_thr_full).astype(int)
    gate_pre_final, gate_model_final = fit_gbdt_classifier_with_progress(
        x_train=X_train,
        y_train=y_gate_full,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        n_estimators=args.n_estimators,
        prefix="Correction gate classifier (final)",
        model_type=args.model_type,
    )

    x_test_corr_t = corr_pre_final.transform(X_test)
    corr_pred_test = corr_model_final.predict(x_test_corr_t)
    x_test_gate_t = gate_pre_final.transform(X_test)
    if hasattr(gate_model_final, "predict_proba"):
        gate_prob_test = gate_model_final.predict_proba(x_test_gate_t)[:, 1]
    else:
        gate_prob_test = gate_model_final.predict(x_test_gate_t).astype(float)

    baseline_test = test_df[baseline_col].map(extract_numeric).to_numpy(dtype=float)
    if use_segment_alpha:
        safe_base_for_bin = baseline_test.copy()
        if np.isnan(safe_base_for_bin).any():
            safe_base_for_bin[np.isnan(safe_base_for_bin)] = np.nanmedian(baseline_train)
        test_bin_idx = assign_bins(safe_base_for_bin, best_alpha_edges)
        test_alpha = best_alpha_by_bin[test_bin_idx]
    else:
        test_alpha = np.full_like(gate_prob_test, best_alpha, dtype=float)

    test_slog_pred = signed_log1p(baseline_test) + test_alpha * gate_prob_test * corr_pred_test
    test_pred = signed_expm1(test_slog_pred)
    valid_test_base = np.abs(baseline_test) > BASELINE_EPS
    test_ratio_pred = np.full_like(test_pred, np.nan, dtype=float)
    test_ratio_pred[valid_test_base] = test_pred[valid_test_base] / baseline_test[valid_test_base]
    test_ratio_pred = np.clip(test_ratio_pred, ratio_lower, ratio_upper)
    test_pred = baseline_test * test_ratio_pred
    missing_test_baseline_rows = int(np.isnan(baseline_test).sum())
    near_zero_test_baseline_rows = int((~np.isnan(baseline_test) & (np.abs(baseline_test) <= BASELINE_EPS)).sum())

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
    print(f"Model type: {args.model_type}")
    print(f"Feature semantics mode: {llm_mode}")
    print(f"LLM profiling used: {'yes' if llm_used else 'no'}")
    print(f"Feature semantic cache used: {'yes' if llm_cache_used else 'no'}")
    print(f"Feature semantic cache file: {semantic_cache_path}")
    print(f"Dropped id-like columns: {llm_summary['dropped_id_like_cols']}")
    print(f"Datetime-expanded columns: {llm_summary['datetime_expanded_cols']}")
    print(f"Year-transformed columns: {llm_summary['year_transformed_cols']}")
    print(f"Alpha candidates: {alpha_candidates}")
    print(f"Best alpha (validation): {best_alpha}")
    print(f"Segmented alpha used: {'yes' if use_segment_alpha else 'no'}")
    print(f"Baseline alpha bins: {args.baseline_alpha_bins}, min bin rows: {args.min_bin_rows}")
    print(f"Tuning objective: {args.tune_objective}")
    print(f"Validation relative gain: {rel_gain:.6f}")
    print(f"Fallback to baseline correction: {'yes' if fallback_to_baseline else 'no'}")
    print(f"Gate quantile selected: {selected_gate_quantile}")
    print(f"Gate threshold |corr_log| (validation/final): {fmt(gate_thr)}/{fmt(gate_thr_full)}")
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
    print(
        "Dropped feature cols (sparse/constant): "
        f"{feature_drop_summary['dropped_sparse_cols']}/"
        f"{feature_drop_summary['dropped_constant_cols']}"
    )
    print(
        "Dropped feature cols (high-unique id-like): "
        f"{feature_drop_summary['dropped_high_unique_id_like_cols']}"
    )
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
    if near_zero_test_baseline_rows > 0:
        print(
            "Rows with near-zero baseline in test: "
            f"{near_zero_test_baseline_rows} (generated becomes NaN for these rows)"
        )
    print()

    print("=== Holdout Metrics (20% validation split) ===")
    print(f"Gate accuracy : {gate_acc_val:.6f}")
    print("---")
    print(f"Baseline MAE  : {fmt(baseline_val_mae)}")
    print(f"Baseline RMSE : {fmt(baseline_val_rmse)}")
    print(f"Baseline R2   : {fmt(baseline_val_r2)}")
    print(f"Baseline TrimmedRMSE90 : {fmt(baseline_val_metrics['trimmed_rmse90'])}")
    print("---")
    print(f"MAE  : {fmt(val_mae)}")
    print(f"RMSE : {fmt(val_rmse)}")
    print(f"R2   : {fmt(val_r2)}")
    print(f"TrimmedRMSE90 : {fmt(val_trimmed_rmse90)}")
    print()
    print("Prediction complete. New file written with generated column.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
