# Conservative Baseline Correction for Tabular Regression

## A Dependency-Aware, Adaptively Gated Correction Framework

**Application domain:** ESG water-consumption prediction (`target`) with a strong rule-based baseline (`wasser_berechnet`).

This document describes the method implemented in `train_and_generate.py` and explains its conceptual relationship to **SAGE** (*Sparse Adaptive Guidance for Dependency-Aware Tabular Data Generation*, ACL 2026).

---

## 1. Problem Setting

We are given tabular records with:

| Symbol | Role |
|--------|------|
| `target` | Ground-truth water consumption (e.g., `esg_firma_esg-bewertung__input__wasserverbrauch-m3`) |
| `baseline` | An existing deterministic or rule-based prediction (e.g., `esg_firma_wasser_berechnet`) |
| `X` | Mixed numeric/categorical features (strings, missing values, noisy numerics) |

**Goal:** Produce `generated` on the test set such that `generated` is closer to `target` than `baseline`, while remaining conservative when the learned correction does not generalize.

**Design choice:** We do **not** predict `target` from scratch. Instead, we learn a **residual correction on top of a strong baseline**, analogous to how SAGE does not generate tabular rows from a fully dense, unguided model but applies **sparse, adaptive guidance** to avoid spurious dependencies.

---

## 2. High-Level Idea

The final prediction is:

```
generated = baseline × clip( ratio_pred , ratio_lower , ratio_upper )
```

where `ratio_pred` is obtained from a signed-log correction pipeline:

```
slog_pred = signed_log1p(baseline) + α · gate_prob · correction_pred
ratio_pred = signed_expm1(slog_pred) / baseline
```

| Component | Role |
|-----------|------|
| `correction_pred` | Regressor predicting the signed-log residual `signed_log1p(target) − signed_log1p(baseline)` |
| `gate_prob` | Classifier probability that a sample **should** be corrected |
| `α` | Global (or segmented) blending strength, tuned on validation |
| `ratio_lower`, `ratio_upper` | Train-derived bounds on `target/baseline`, preventing extreme corrections |

This is a **three-level adaptation**:
1. **Whether** to correct (`gate_prob`)
2. **How much** to correct (`α`, optionally per baseline bin)
3. **How far** correction may deviate from baseline (`ratio` clipping)

---

## 3. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Data ingestion & semantic feature engineering                │
│ 2. Sample cleaning (positive target/baseline, valid ratios)     │
│ 3. Feature preprocessing, filtering, optional pruning             │
│ 4. Test-distribution importance weighting (leak-safe)           │
│ 5. OOF / holdout tuning: correction model + gate + α + q          │
│ 6. Conservative fallback check                                  │
│ 7. Optional post-hoc calibration (validation-only)              │
│ 8. Full-data retraining & test prediction                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Step-by-Step Method

### 4.1 Semantic Feature Engineering

**What it does**
- Classifies columns as ID-like, datetime, year-like, numeric, or categorical.
- Uses a local LLM (Qwen) with caching, or heuristic rules as fallback.

**Actions**
| Column type | Treatment |
|-------------|-----------|
| ID-like | Dropped |
| Datetime | Expanded into year, month, day, hour, weekday |
| Year-like | Kept + `age_from_2026` transform |
| Other | Retained |

**Why:** Raw tables mix heterogeneous semantics. Treating years as ordinals or IDs as numeric features introduces spurious dependencies—precisely the failure mode SAGE addresses via sparse, value-aware pseudo-features.

---

### 4.2 Sample Cleaning

Training rows are kept only if:
- `target > 0` and `baseline > 0`
- `target/baseline` is finite and within float32-safe range

Non-positive or invalid rows are treated as noise and excluded from all training and tuning.

---

### 4.3 Feature Preprocessing

**Numeric pipeline**
- Median imputation + missing indicators
- Quantile clipping (1st–99th percentile)

**Categorical pipeline**
- Most-frequent imputation + missing indicators
- Ordinal encoding (unknown → −1)

**Filtering**
- Drop sparse columns (non-null ratio < 1%)
- Drop constant columns
- Drop high-unique ID-like columns (unique ratio ≥ 98% with ID name hints)
- Optional auto-pruning: if feature count > 500, keep top features by \|Spearman(target)\|

---

### 4.4 Signed-Log Residual Space

Corrections are learned in signed-log space:

```
signed_log1p(x) = sign(x) · log(1 + |x|)
signed_expm1(x) = sign(x) · (exp(|x|) − 1)
```

**Training target for the regressor:**
```
y_corr = signed_log1p(target) − signed_log1p(baseline)
```

**Rationale:** Water consumption spans large magnitudes with heavy tails. Signed-log stabilizes optimization, reduces sensitivity to outliers, and keeps corrections well-behaved near zero.

---

### 4.5 Ratio Clipping (Fidelity Constraint)

From valid training samples, compute `ratio = target / baseline` and set:

```
ratio_lower = Q_{(1−c)/2}(ratio)
ratio_upper = Q_{1−(1−c)/2}(ratio)
```

Default coverage `c = 0.93` (central 93% of training ratios).

**Role:** Acts as a **domain constraint**—similar in spirit to SAGE's emphasis on reducing constraint violations in synthetic generation. Here we bound how far `generated` may deviate from `baseline` in multiplicative terms.

---

### 4.6 Correction Regressor

A tree-based regressor (HistGradientBoosting by default) predicts `y_corr` from `X`.

**Supported backends:** GBDT (`--model-type gbdt`), HistGradientBoosting (`--model-type hgbt`, default).

**Validation:** K-fold Out-of-Fold (OOF, default 3 folds) or 20% holdout.

Each fold trains on a subset and predicts OOF corrections for the held-out fold, yielding unbiased validation estimates for downstream tuning.

---

### 4.7 Gate Classifier (Sparse Correction Selection)

For each candidate gate quantile `q ∈ {0.4, 0.5, 0.6, 0.7}`:

1. **Threshold:** `τ_q = quantile(|y_corr_train|, q)`
2. **Label:** `y_gate = 1` if `|y_corr| ≥ τ_q`, else `0`
3. **Train** a binary classifier → `gate_prob ∈ [0, 1]`

**Interpretation:** The gate decides **which samples merit correction**. Low-residual samples (baseline already adequate) receive low `gate_prob` and are left mostly unchanged—**sparse application** of the correction mechanism.

**Degenerate case:** If all labels are one class, skip classifier training and use a constant prediction.

---

### 4.8 Alpha Blending (Adaptive Correction Strength)

For each candidate `α` in `{0, 0.4, 0.6, 0.8, 1.0}`:

```
slog_pred = signed_log1p(baseline) + α · gate_prob · correction_pred
generated_candidate = baseline × clip(signed_expm1(slog_pred) / baseline)
```

Select `α` minimizing the validation objective (default: **MAE**).

**Optional segmented α:** When enabled, `α` is tuned per baseline quantile bin; otherwise a single global `α` is used.

---

### 4.9 Hyperparameter Selection (q, α, segmented vs global)

For each `q`:
1. Train gate classifier (OOF per fold)
2. Pick best `α` (and optionally segmented `α`)
3. Score on validation with weighted objective
4. Apply **gate-rate penalty** if mean `gate_prob` falls outside `[0.10, 0.55]`—discouraging over- or under-correction

Select the `q` with lowest penalized validation score.

**Tuning objectives:** `mae` | `rmse` | `trimmed_rmse90` | `hybrid` (= 0.8·MAE + 0.2·TrimmedRMSE90)

---

### 4.10 Test-Distribution Importance Weighting (Leak-Safe)

To narrow the validation–test gap without using test labels:

1. Train a domain classifier: `P(test | X)` on concatenated train+test features
2. Assign each training sample weight `w ∝ P(test|X) / (1 − P(test|X))`, clipped and normalized
3. Use `w` in validation metric computation (MAE, etc.)

**Only test features are used—never test `target`.** This reweights validation toward regions of feature space that resemble the test distribution.

---

### 4.11 Conservative Fallback

Compute relative validation gain over baseline:

```
rel_gain = (objective_baseline − objective_selected) / objective_baseline
```

If `rel_gain < min_validation_rel_gain` (default 0.2%), set `α = 0` and output `generated = baseline` (after ratio handling).

**Purpose:** Do not deploy corrections that fail to beat baseline on validation—a **safety valve** analogous to SAGE's logit smoothing when context is uninformative (avoid overconfident, harmful updates).

---

### 4.12 Optional Post-Hoc Calibration

When enabled (default: off), fit a calibrator on **validation predictions only**:

| Method | Description |
|--------|-------------|
| `segmented_bias` | Per baseline bin: `bias_b = mean(target − pred)`; apply `pred + bias_b` |
| `isotonic` | Monotonic mapping pred → target |
| `linear` | Linear regression pred → target |

Applied to test only if validation RMSE gain exceeds `calibration_min_rel_gain`.

---

### 4.13 Final Training and Test Inference

1. Retrain correction regressor on all valid training data
2. Retrain gate classifier with selected `q`
3. Predict `correction_pred` and `gate_prob` on test
4. Apply selected `α` (global or segmented)
5. Apply ratio clipping; optional calibration
6. Write `generated.csv`

Rows with near-zero `baseline` on test → `generated = NaN`.

---

## 5. Conceptual Connection to SAGE (ACL 2026)

SAGE and our method address **different tasks** (synthetic tabular **generation** vs. baseline **correction** for regression), but they share a common philosophy: **do not treat all feature–target dependencies equally; adapt guidance sparsely and conservatively based on context.**

### 5.1 Summary Comparison

| Theme | SAGE | Our Method |
|-------|------|------------|
| **Core problem** | LLM tabular generation suffers from dense attention and static dependency graphs | Direct residual prediction overfits; baseline is strong but imperfect |
| **Sparsity** | Feature Selector: keep only pseudo-features with MI > τ | Gate: correct only when \|residual\| is large; feature pruning drops weak/ID columns |
| **Adaptivity** | Dependencies are value-conditioned (e.g., Loan Purpose → Age varies by purpose) | Segmented α/calibration by baseline bin; gate_prob varies per sample |
| **Strength modulation** | Logit Correction: `z' = z·(1 + λ·Δ)`; sharpen when informative, smooth when weak | `α · gate_prob · correction`; α tuned globally/segmented; penalty on extreme gate rates |
| **Constraints** | Reduce constraint violation rate in synthetic data | Ratio clipping bounds `generated/baseline`; fallback to baseline |
| **Preprocessing** | Pseudo-feature discretization + MI matrix | Semantic typing (LLM/heuristic), datetime/year expansion, correlation pruning |
| **Validation signal** | Downstream utility on synthetic→real | OOF + optional test-distribution reweighting |

---

### 5.2 Sparse Guidance ↔ Gate + Feature Pruning

**SAGE (Feature Selector):**  
Build a sparse MI-based dependency graph; at generation time, condition the LLM only on pseudo-features whose MI with the target exceeds threshold τ. This avoids dense, spurious correlations.

**Our method (Gate classifier):**  
At prediction time, apply the learned correction only when `gate_prob` is high—i.e., when the sample's residual magnitude suggests baseline is likely inadequate. Samples where baseline is already close to target receive minimal intervention.

**Our method (Feature pruning):**  
Auto-prune and drop ID-like/sparse features by correlation with `target`, reducing the regressor's exposure to irrelevant or high-cardinality columns—another form of **sparse, relevance-driven** feature use.

> **Shared principle:** *Selective use of information beats dense, uniform application.*

---

### 5.3 Adaptive Strength ↔ Alpha and Gate Probability

**SAGE (Logit Correction):**  
Measure how informative the current prefix is relative to training average:

```
Δ = μ_sample / μ_train − 1
z' = z · (1 + λ · Δ)
```

When Δ > 0 (informative context), sharpen generation; when Δ < 0 (weak context), smooth and avoid overconfident outputs.

**Our method:**  
Correction strength is **explicitly modulated**:

```
effective_correction = α · gate_prob · correction_pred
```

- `gate_prob` ≈ SAGE's sample-level informativeness (is this row worth correcting?)
- `α` ≈ global/segmented λ (how aggressively to apply corrections that pass the gate)
- `ratio` clipping ≈ hard bounds when modulation would violate plausibility

> **Shared principle:** *Adapt the influence of the learned component to context confidence; back off when evidence is weak.*

---

### 5.4 Value-Conditioned Dependencies ↔ Segmented Strategies

**SAGE:**  
Emphasizes that dependencies are **value-conditioned**—e.g., the relationship between Loan Purpose and Age depends on which loan purpose is active. Static graphs miss this.

**Our method:**  
Optional **segmented α** and **segmented bias calibration** partition the sample space by `baseline` magnitude. Different baseline ranges may require different correction strengths—mirroring value-conditioned adaptation, but along the baseline axis rather than arbitrary feature crosses.

**Test-distribution weighting** further adapts validation to regions of `X` that resemble test, addressing **distribution-conditioned** generalization without test labels.

---

### 5.5 Fidelity and Safety ↔ Ratio Clipping and Fallback

**SAGE:**  
Reports constraint violation rates (e.g., geographic bounds in CA Housing) and aims to preserve realistic feature dependencies in synthetic rows.

**Our method:**  
- **Ratio clipping** enforces that `generated/baseline` stays within a train-derived plausible band.
- **Conservative fallback** reverts to baseline when validation gain is insufficient.
- **Gate-rate penalty** prevents pathological regimes where nearly all or almost no samples are corrected.

> **Shared principle:** *Learned guidance must respect domain plausibility; when it does not help, do not force it.*

---

### 5.6 Preprocessing Philosophy

**SAGE** discretizes numerics and one-hot encodes categoricals into pseudo-features to estimate MI—a **dependency-aware representation**.

**Our method** uses LLM/heuristic **semantic typing** to transform raw columns into modeling-friendly forms (datetime parts, year age, dropped IDs). Both approaches recognize that **raw tabular encoding hides structure** and that explicit preprocessing is needed before a model can exploit meaningful dependencies.

---

## 6. What Differs from SAGE (Scope and Mechanism)

| Aspect | SAGE | Our Method |
|--------|------|------------|
| Task | Generative modeling (synthetic rows) | Discriminative correction (regression) |
| Model | LLM autoregressive generation | Tree models (HGBT / GBDT) |
| Guidance mechanism | MI matrix + prefix filtering / logit scaling | Gate classifier + α blending + ratio bounds |
| Supervision | Unsupervised / generative training | Supervised residual on `target` with strong baseline |
| Output | Full synthetic table | Single corrected column `generated` |

The connection is **conceptual and methodological**, not a direct implementation of SAGE. Our pipeline can be viewed as applying SAGE-like ideas—**sparse, adaptive, constraint-aware guidance**—to the complementary problem of **improving a strong baseline predictor** rather than generating synthetic data.

---

## 7. Evaluation Protocol

Primary script: `measure_error.py`

- Compares `generated` vs `baseline` against `target` on the same filtered rows (positive target/baseline, valid numerics).
- Metrics: MAE, RMSE, MedianAE, P90/P95 AE, TrimmedMAE90, TrimmedRMSE90, Bias, WAPE, R², etc.

Diagnostic scripts:
- `visualize_prediction_advantage.py` — advantage plots and heatmaps
- `analyze_model_diagnostics.py` — segmented performance
- `analyze_feature_signal_drift.py` — feature signal and train–test drift

---

## 8. Default Configuration (Current)

| Parameter | Default | Role |
|-----------|---------|------|
| `model_type` | `hgbt` | Backend for correction & gate |
| `n_estimators` | `300` | Boosting iterations (HGBT/GBDT) |
| `tune_objective` | `mae` | Validation optimization target |
| `ratio_coverage` | 0.93 | Central ratio clip band |
| `gate_quantile_candidates` | 0.4–0.7 | Gate threshold search |
| `alpha_grid` | 0, 0.4, 0.6, 0.8, 1.0 | Blending search |
| `enable_segmented_alpha` | false | Global α by default |
| `enable_test_distribution_weighting` | true | Leak-safe val reweighting |
| `oof_folds` | 3 | OOF validation |
| `enable_posthoc_calibration` | false | Off by default |

---

## 9. One-Sentence Summary

We improve a strong water-consumption baseline through **sparse, adaptively gated, signed-log residual correction** with **train-derived ratio constraints** and **leak-safe test-aware validation**—a regression correction framework that shares SAGE's core belief that tabular learning should be **dependency-aware, selectively guided, and conservative when context does not support confident updates**.

---

## References

- Yang, S., Zhang, Z., Prenkaj, B., & Kasneci, G. (2026). *SAGE: Sparse Adaptive Guidance for Dependency-Aware Tabular Data Generation.* ACL 2026. Code: github.com/ShuoYangtum/SAGE
