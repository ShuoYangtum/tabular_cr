#!/usr/bin/env python3
"""
Leak-safe post-hoc calibration for prediction column.

Two-step workflow:
1) fit   : fit calibrator on a labeled calibration set and save calibrator params
2) apply : apply saved calibrator to another file without reading target labels
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression


DEFAULT_FIT_FILE = "generated.csv"
DEFAULT_APPLY_FILE = "generated.csv"
DEFAULT_TARGET_COL = "target"
DEFAULT_PRED_COL = "generated"
DEFAULT_OUTPUT_FILE = "generated_calibrated.csv"
DEFAULT_METHOD = "isotonic"
DEFAULT_CALIBRATOR_FILE = "calibrator.json"

NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
MISSING = {"", "nan", "none", "null", "na", "n/a", "-", "--", "unknown"}


def extract_numeric(value) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return np.nan
        return v
    s = str(value).strip()
    if not s:
        return np.nan
    s = s.replace(",", "")
    s = re.sub(r"\s+", "", s)
    if s.lower() in MISSING:
        return np.nan
    m = NUMERIC_PATTERN.search(s)
    if not m:
        return np.nan
    try:
        v = float(m.group())
        if math.isnan(v) or math.isinf(v):
            return np.nan
        return v
    except Exception:
        return np.nan


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot) if ss_tot > 0 else np.nan
    bias = float(np.mean(err))
    return {"mae": mae, "rmse": rmse, "r2": r2, "bias": bias}


def fit_calibrator(p_fit: np.ndarray, y_fit: np.ndarray, method: str) -> tuple[dict, np.ndarray]:
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(p_fit, y_fit)
        p_fit_cal = calibrator.predict(p_fit)
        payload = {
            "method": "isotonic",
            "x_thresholds": calibrator.X_thresholds_.tolist(),
            "y_thresholds": calibrator.y_thresholds_.tolist(),
            "x_min": float(np.min(calibrator.X_thresholds_)),
            "x_max": float(np.max(calibrator.X_thresholds_)),
        }
        return payload, p_fit_cal

    calibrator = LinearRegression()
    calibrator.fit(p_fit.reshape(-1, 1), y_fit)
    p_fit_cal = calibrator.predict(p_fit.reshape(-1, 1))
    payload = {
        "method": "linear",
        "coef": float(calibrator.coef_[0]),
        "intercept": float(calibrator.intercept_),
    }
    return payload, p_fit_cal


def apply_calibrator_to_values(values: np.ndarray, payload: dict) -> np.ndarray:
    method = payload.get("method")
    if method == "isotonic":
        x = np.asarray(payload["x_thresholds"], dtype=float)
        y = np.asarray(payload["y_thresholds"], dtype=float)
        if x.ndim != 1 or y.ndim != 1 or len(x) < 2 or len(x) != len(y):
            raise ValueError("Invalid isotonic calibrator payload.")
        x_clip = np.clip(values, float(payload.get("x_min", np.min(x))), float(payload.get("x_max", np.max(x))))
        return np.interp(x_clip, x, y)
    if method == "linear":
        coef = float(payload["coef"])
        intercept = float(payload["intercept"])
        return coef * values + intercept
    raise ValueError(f"Unsupported calibrator method: {method}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Leak-safe post-hoc calibrator (fit/apply modes).")
    parser.add_argument("--mode", choices=["fit", "apply"], required=True, help="Run mode")
    parser.add_argument("--fit-file", default=DEFAULT_FIT_FILE, help="Labeled file used to fit calibrator")
    parser.add_argument("--apply-file", default=DEFAULT_APPLY_FILE, help="File to apply calibrated predictions")
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL, help="Target column name in fit file")
    parser.add_argument("--pred-col", default=DEFAULT_PRED_COL, help="Prediction column to calibrate")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE, help="Output calibrated file")
    parser.add_argument(
        "--calibrator-file",
        default=DEFAULT_CALIBRATOR_FILE,
        help="Saved calibrator JSON file path",
    )
    parser.add_argument("--method", choices=["isotonic", "linear"], default=DEFAULT_METHOD, help="Calibration method")
    args = parser.parse_args()
    calibrator_path = Path(args.calibrator_file)

    if args.mode == "fit":
        fit_path = Path(args.fit_file)
        if not fit_path.exists():
            raise FileNotFoundError(fit_path)
        fit_df = (
            pd.read_csv(fit_path, low_memory=False) if fit_path.suffix.lower() == ".csv" else pd.read_excel(fit_path)
        )
        for c in [args.target_col, args.pred_col]:
            if c not in fit_df.columns:
                raise KeyError(f"fit file missing column: {c}")

        y = fit_df[args.target_col].map(extract_numeric)
        p = fit_df[args.pred_col].map(extract_numeric)
        mask = y.notna() & p.notna()
        if not mask.any():
            raise ValueError("No valid fit rows after cleaning.")

        y_fit = y[mask].to_numpy(dtype=float)
        p_fit = p[mask].to_numpy(dtype=float)
        payload, p_fit_cal = fit_calibrator(p_fit, y_fit, args.method)

        meta = {
            "pred_col": args.pred_col,
            "target_col": args.target_col,
            "fit_file": str(fit_path),
            "rows_used": int(mask.sum()),
            "payload": payload,
        }
        calibrator_path.parent.mkdir(parents=True, exist_ok=True)
        calibrator_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        before = metrics(y_fit, p_fit)
        after = metrics(y_fit, p_fit_cal)
        print("=== Calibrator fit summary ===")
        print(f"Method: {payload['method']}")
        print(f"Fit rows used: {int(mask.sum())}")
        print(
            f"Before MAE/RMSE/R2/Bias: "
            f"{before['mae']:.6f} / {before['rmse']:.6f} / {before['r2']:.6f} / {before['bias']:.6f}"
        )
        print(
            f"After  MAE/RMSE/R2/Bias: "
            f"{after['mae']:.6f} / {after['rmse']:.6f} / {after['r2']:.6f} / {after['bias']:.6f}"
        )
        print(f"Saved calibrator: {calibrator_path}")
        return 0

    # apply mode
    apply_path = Path(args.apply_file)
    if not apply_path.exists():
        raise FileNotFoundError(apply_path)
    if not calibrator_path.exists():
        raise FileNotFoundError(f"Calibrator file not found: {calibrator_path}")

    meta = json.loads(calibrator_path.read_text(encoding="utf-8"))
    payload = meta.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("Invalid calibrator file: missing payload.")

    apply_df = (
        pd.read_csv(apply_path, low_memory=False) if apply_path.suffix.lower() == ".csv" else pd.read_excel(apply_path)
    )
    if args.pred_col not in apply_df.columns:
        raise KeyError(f"apply file missing column: {args.pred_col}")

    p_apply = apply_df[args.pred_col].map(extract_numeric)
    valid_apply = p_apply.notna()
    calibrated = np.full(len(apply_df), np.nan, dtype=float)
    if valid_apply.any():
        x = p_apply[valid_apply].to_numpy(dtype=float)
        calibrated_vals = apply_calibrator_to_values(x, payload)
        calibrated[valid_apply.to_numpy()] = calibrated_vals

    out_df = apply_df.copy()
    out_df[f"{args.pred_col}_calibrated"] = calibrated
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("=== Calibrator apply summary ===")
    print(f"Method: {payload.get('method')}")
    print(f"Rows calibrated: {int(valid_apply.sum())}")
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
