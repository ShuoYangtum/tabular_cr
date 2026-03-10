#!/usr/bin/env python3
"""
LLM-assisted feature profiling for tabular columns.

It uses a local Qwen model folder (default: ./Qwen3-4B-Instruct-2507)
to classify columns such as:
- id-like
- datetime-like
- year-like
- numeric
- categorical
- text-like

It also outputs recommended handling strategies for modeling.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# =========================
# Editable defaults
# =========================
DEFAULT_TRAIN_FILE = "../modified/train_3_clean.csv"
DEFAULT_TEST_FILE = "../modified/test_3_clean.csv"
DEFAULT_MODEL_PATH = "/data/models/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT_JSON = "feature_profile_report.json"
DEFAULT_SAMPLE_SIZE = 20
DEFAULT_MAX_NEW_TOKENS = 220
DEFAULT_TEMPERATURE = 0.0


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}. Use .csv/.xlsx/.xls")


def sample_values(series: pd.Series, sample_size: int) -> List[str]:
    s = series.dropna()
    if s.empty:
        return []
    # Convert to string and keep unique examples.
    values = s.astype(str).drop_duplicates().head(sample_size).tolist()
    return values


def quick_stats(series: pd.Series) -> Dict[str, Any]:
    total = int(series.shape[0])
    na_count = int(series.isna().sum())
    nunique = int(series.nunique(dropna=True))
    uniq_ratio = (nunique / max(total - na_count, 1)) if total > 0 else 0.0

    as_num = pd.to_numeric(series, errors="coerce")
    numeric_ratio = float(as_num.notna().mean()) if total > 0 else 0.0
    datetime_ratio = float(pd.to_datetime(series, errors="coerce", utc=True).notna().mean()) if total > 0 else 0.0

    year_like_ratio = 0.0
    if numeric_ratio > 0:
        y = as_num.dropna()
        if not y.empty:
            year_like_ratio = float(((y >= 1900) & (y <= 2100) & (np.floor(y) == y)).mean())

    return {
        "total": total,
        "na_count": na_count,
        "nunique": nunique,
        "unique_ratio_non_na": round(float(uniq_ratio), 6),
        "numeric_parse_ratio": round(float(numeric_ratio), 6),
        "datetime_parse_ratio": round(float(datetime_ratio), 6),
        "year_like_ratio_among_numeric": round(float(year_like_ratio), 6),
    }


def build_prompt(col_name: str, stats: Dict[str, Any], train_samples: List[str], test_samples: List[str]) -> str:
    schema = {
        "column_name": col_name,
        "final_type": "one of [id, datetime, year, numeric, categorical, text, unknown]",
        "confidence": "0~1 float",
        "reason": "short reason",
        "recommended_actions": {
            "drop_as_feature": "true/false",
            "parse_datetime": "true/false",
            "extract_datetime_parts": "list like [year,month,day,hour,weekday] or []",
            "treat_as_year": "true/false",
            "treat_as_numeric": "true/false",
            "treat_as_categorical": "true/false",
            "notes": "short handling note",
        },
    }

    return (
        "You are a tabular feature engineering expert.\n"
        "Classify the following column and return STRICT JSON only.\n"
        "No markdown, no extra text.\n\n"
        f"JSON schema example:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Column name: {col_name}\n"
        f"Quick stats: {json.dumps(stats, ensure_ascii=False)}\n"
        f"Train samples: {json.dumps(train_samples, ensure_ascii=False)}\n"
        f"Test samples: {json.dumps(test_samples, ensure_ascii=False)}\n"
    )


def safe_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()
    # Try direct parse.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try extracting first {...} JSON block.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return {
        "final_type": "unknown",
        "confidence": 0.0,
        "reason": "failed to parse model output as JSON",
        "recommended_actions": {
            "drop_as_feature": False,
            "parse_datetime": False,
            "extract_datetime_parts": [],
            "treat_as_year": False,
            "treat_as_numeric": False,
            "treat_as_categorical": True,
            "notes": "fallback: keep as categorical or inspect manually",
        },
        "raw_model_output": text[:1000],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile tabular features using local Qwen model.")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE, help="Train table path")
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE, help="Test table path")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local Qwen model directory")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON, help="Output report JSON path")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Sample values per column")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="LLM max_new_tokens")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="LLM sampling temperature")
    args = parser.parse_args()

    train_path = Path(args.train_file)
    test_path = Path(args.test_file)
    model_path = Path(args.model_path)
    output_path = Path(args.output_json)

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    train_df = read_table(train_path)
    test_df = read_table(test_path)
    train_df.columns = [str(c).strip() for c in train_df.columns]
    test_df.columns = [str(c).strip() for c in test_df.columns]

    # Load model/tokenizer lazily only once.
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

    all_cols = sorted(set(train_df.columns) | set(test_df.columns))
    results: List[Dict[str, Any]] = []

    for col in tqdm(all_cols, desc="LLM inferring columns", unit="col"):
        train_series = train_df[col] if col in train_df.columns else pd.Series(dtype="object")
        test_series = test_df[col] if col in test_df.columns else pd.Series(dtype="object")

        stats_train = quick_stats(train_series)
        stats_test = quick_stats(test_series)
        combined_stats = {
            "train": stats_train,
            "test": stats_test,
        }
        train_samples = sample_values(train_series, args.sample_size)
        test_samples = sample_values(test_series, args.sample_size)

        prompt = build_prompt(col, combined_stats, train_samples, test_samples)
        messages = [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        parsed = safe_json_from_text(generated_text)

        result_item = {
            "column": col,
            "stats": combined_stats,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "llm_decision": parsed,
        }
        results.append(result_item)

    report = {
        "train_file": str(train_path),
        "test_file": str(test_path),
        "model_path": str(model_path),
        "column_count": len(all_cols),
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== LLM Feature Profiling Done ===")
    print(f"Columns analyzed: {len(all_cols)}")
    print(f"Output report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
