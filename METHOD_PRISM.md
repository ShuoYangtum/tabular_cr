# PRISM Prediction Tool Guide

**PRISM** (Prior-anchored Robust quantile modeling with conformal trust Intervals for Skewed Measurements)  
A tabular prediction tool for ESG enterprise resource consumption (water use, electricity use, and similar metrics).


| Item                | Description                                                        |
| ------------------- | ------------------------------------------------------------------ |
| Entry point         | `train_prism.py`                                                   |
| Optional evaluation | `measure_error.py` (requires ground-truth labels in the test file) |
| Delivery form       | Single-script package; no external LLM or GPU service required     |


---

## 1. Purpose

PRISM produces batch predictions for company-level ESG indicators when:

- feature quality is mixed and incomplete,
- targets span many orders of magnitude,
- a small share of extreme cases dominates total error.

Unlike a generic regressor, PRISM:

- **With a business baseline**: applies a **limited, guarded correction** so typical firms are not made worse while chasing overall MAE.
- **Without a baseline**: **builds its own prior anchor** from categorical structure in the data.
- **Never uses test-set labels** during training or tuning. The test file supplies features (and an optional baseline) for inference only.

---

## 2. Quick Start

### 2.1 Install

```bash
pip install numpy pandas scikit-learn
```

Place `train_prism.py` (and optionally `measure_error.py`) in your project folder with `train_clean.csv` and `test_clean.csv`.

### 2.2 Configure the task (one-time)

Open `train_prism.py` and edit the **Defaults** block at the top:

```python
DEFAULT_TARGET_COL = "your_target_column"
DEFAULT_BASELINE_COL = ""          # leave "" if no baseline
LEAKAGE_FEATURES = { ... }         # columns that must not be used as features
```

Alternatively, pass the same values on the command line (see Section 6).

### 2.3 Run prediction

**Electricity (no baseline):**

```bash
python train_prism.py \
  --train-file train_clean.csv \
  --test-file test_clean.csv \
  --target-col esg_firma_esg-bewertung__input__elektrizitaetsverbrauch-kwh
```

**Water (with baseline):**

```bash
python train_prism.py \
  --train-file train_clean.csv \
  --test-file test_clean.csv \
  --target-col esg_firma_esg-bewertung__input__wasserverbrauch-m3 \
  --baseline-col esg_firma_wasser_berechnet
```

### 2.4 Check outputs

| File | What to look at |
|------|-----------------|
| `generated.csv` | `generated`, `generated_low`, `generated_high`, `generated_trust`, `generated_tail_risk` |
| `prism_report.json` | `fallback_to_anchor`, `selected` (λ, tail settings), OOF metrics |
| Console | `fallback_to_anchor=False` means a model correction was deployed |

### 2.5 Evaluate (optional, labels required)

```bash
python measure_error.py --file generated.csv
```

Use the same `--target-col` / `--generated-col` as in training if they differ from script defaults.

---

## 3. Our contributions

The following are the main methodological and engineering outcomes of this work for the ESG prediction setting.

### 3.1 Anchor-constrained prediction

Predictions are expressed as **anchor + controlled deviation**:

- Water: the rule-based baseline is the anchor.
- Electricity: a hierarchical empirical-Bayes prior (shrunk group medians over selected categorical fields such as industry) is the anchor.

In log space, the final estimate shrinks toward the anchor by a coefficient λ. λ is chosen by cross-validation on the training set. **If no candidate reliably beats the anchor, the tool falls back to the anchor.**  
This design addresses a failure mode observed with earlier residual-correction pipelines: overall error metrics could look acceptable while MedianAE on typical firms degraded sharply.

### 3.2 Robust modeling for heavy-tailed targets

Targets are transformed with `log1p` and modeled with **quantile Gradient Boosting** (5% / 50% / 75% / 90% / 95%). The median serves typical firms; a separate tail-risk score identifies likely high consumers and can switch those rows to an upper quantile.  

Relative to the previous stack (residual correction + gate + multi-α + ratio clipping), deployment is reduced to **anchor shrinkage + optional tail switch + two hard guards**, which is easier to maintain and reproduce.

### 3.3 Deployment guards

When a real baseline is available, cross-validation enforces:


| Guard                 | Default rule                                                        | Role                                               |
| --------------------- | ------------------------------------------------------------------- | -------------------------------------------------- |
| MedianAE guard        | Candidate MedianAE may not exceed baseline MedianAE by more than 5% | Protects quality on the majority of firms          |
| MAE guard             | Same for MAE                                                        | Blocks overall error regression                    |
| Minimum relative gain | Less than 0.5% gain vs. anchor → fallback                           | Avoids deploying changes with no practical benefit |




### 3.4 Actionable auxiliary outputs

Each test row includes:


| Column                             | Meaning                                            | Business use                                 |
| ---------------------------------- | -------------------------------------------------- | -------------------------------------------- |
| `generated`                        | Point prediction                                   | Primary result                               |
| `generated_low` / `generated_high` | Conformal ~90% interval                            | Risk bounds and audit trail                  |
| `generated_trust`                  | Trust score in [0, 1] (narrower interval → higher) | Prioritize low-trust rows for review         |
| `generated_tail_risk`              | Probability of a high-consumption tail case        | Flag firms that may dominate aggregate error |




### 3.5 One pipeline for both task types

Water (with baseline) and electricity (without baseline) share the same code path. Leakage columns are configured at the top of the script and excluded during training. Feature cleaning uses heuristics (type inference, datetime expansion, ID removal) and **does not depend on an LLM**.

---



## 4. Scope and data requirements

### 4.1 Suitable use cases

- Tabular data with positive continuous targets;
- Training CSV with a target column; test CSV with matching feature columns (target on test is optional and unused);
- Recommended: at least 50 valid training rows (enforced by the tool).



### 4.2 Inputs


| File         | Required columns            | Notes                                 |
| ------------ | --------------------------- | ------------------------------------- |
| Training set | Target + features           | Used for fitting and OOF tuning       |
| Test set     | Features                    | Used for inference; optional baseline |
| Leakage list | Configured in script or CLI | Excluded from model features          |




### 4.3 Outputs


| File                          | Content                                                            |
| ----------------------------- | ------------------------------------------------------------------ |
| `generated.csv` (default)     | Test table plus prediction, interval, trust, and tail-risk columns |
| `prism_report.json` (default) | Tuning choices, OOF metrics, stratified diagnostics                |
| `prism_oof.csv` (default)     | Training OOF predictions for internal analysis                     |


---



## 5. How to run

### 5.1 Environment

```text
Python 3.9+
numpy, pandas, scikit-learn
```



### 5.2 Configuration

Edit the **Defaults** section at the top of `train_prism.py`:

```python
DEFAULT_TARGET_COL = "..."      # target column name
DEFAULT_BASELINE_COL = "..."    # baseline column; use "" if none
LEAKAGE_FEATURES = { ... }      # leakage feature set
```



### 5.3 Examples

**Water (rule-based baseline available):**

```bash
python train_prism.py \
  --train-file train_clean.csv \
  --test-file test_clean.csv \
  --target-col esg_firma_esg-bewertung__input__wasserverbrauch-m3 \
  --baseline-col esg_firma_wasser_berechnet
```

**Electricity (no baseline → synthesized prior anchor):**

```bash
python train_prism.py \
  --train-file train_clean.csv \
  --test-file test_clean.csv \
  --target-col esg_firma_esg-bewertung__input__elektrizitaetsverbrauch-kwh
```



### 5.4 Offline evaluation (optional)

When the test file contains ground truth:

```bash
python measure_error.py --file generated.csv
```

If no baseline column is present, `measure_error.py` reports metrics for `generated` only. By default it reports primary metrics after a fair trim of the worst 5% of rows.

---

## 6. Command-line reference

All flags accept values on the command line or fall back to the **Defaults** block in each script. Paths may be relative or absolute. Column names are case-sensitive strings that must exist in the CSV header.

### 6.1 `train_prism.py`

#### I/O and columns

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--train-file` | `str` (file path) | `train_clean.csv` | Training CSV. Must exist. |
| `--test-file` | `str` (file path) | `test_clean.csv` | Test CSV. Must exist. |
| `--output-file` | `str` (file path) | `generated.csv` | Output CSV with predictions. |
| `--target-col` | `str` | see script | Target column in **train** file (required there). |
| `--baseline-col` | `str` | `""` (empty) | Baseline column name. Empty string → no baseline; prior anchor is synthesized. If set but missing in train, a warning is printed and no-baseline mode is used. |
| `--generated-col` | `str` | `generated` | Name of the point-prediction column. Interval/trust columns use this as a prefix (`{col}_low`, etc.). |
| `--leakage-features` | `str` | see `LEAKAGE_FEATURES` | Comma-separated column names excluded from model features. Use `""` to disable. Derived columns (`col__year`, etc.) matching a root name are excluded too. |
| `--report-file` | `str` (file path) | `prism_report.json` | JSON report path. |
| `--oof-file` | `str` (file path) | `prism_oof.csv` | Training OOF diagnostic CSV. |

#### Model and cross-validation

| Flag | Type | Default | Valid range / choices | Description |
|------|------|---------|----------------------|-------------|
| `--folds` | `int` | `5` | integer ≥ 2 | Cross-validation folds for OOF tuning and prior construction. |
| `--n-estimators` | `int` | `400` | positive integer | Trees per quantile regressor and tail classifier. |
| `--learning-rate` | `float` | `0.06` | positive float | HistGradientBoosting learning rate. |
| `--seed` | `int` | `42` | integer | Random seed for folds and models. |

#### Tuning objective and guards

| Flag | Type | Default | Valid range / choices | Description |
|------|------|---------|----------------------|-------------|
| `--objective` | `str` | `log_mae` | `mae`, `median_ae`, `wape`, `log_mae`, `trimmed_mae90` | Metric minimized when selecting shrinkage λ and tail settings on OOF predictions. |
| `--min-rel-gain` | `float` | `0.005` | float ≥ 0 | Minimum **relative** OOF improvement vs. anchor on `--objective` before deploying the model (when a real baseline exists). Otherwise fallback to anchor. |
| `--median-guard-ratio` | `float` | `1.05` | float; `≤ 0` disables | Reject any candidate whose OOF MedianAE exceeds anchor MedianAE × this factor (baseline mode only). |
| `--mae-guard-ratio` | `float` | `1.05` | float; `≤ 0` disables | Same for MAE. |

#### Prior anchor (no-baseline mode)

| Flag | Type | Default | Valid range / choices | Description |
|------|------|---------|----------------------|-------------|
| `--prior-levels` | `int` | `3` | non-negative integer | Max number of categorical columns in the hierarchical EB prior. `0` → global median only. |
| `--prior-shrinkage` | `float` | `20.0` | float > 0 | Pseudo-count *m* for shrinking group medians toward the parent level. |
| `--disable-prior-anchor` | flag | off | present / absent | If set, do not build the EB prior; use the global training median as anchor when no baseline column is available. |

#### Tail handling

| Flag | Type | Default | Valid range / choices | Description |
|------|------|---------|----------------------|-------------|
| `--tail-quantile` | `float` | `0.9` | float in (0, 1) | Training-target quantile above which a row is labeled “tail” for the risk classifier. |
| `--disable-tail-switch` | flag | off | present / absent | If set, never switch high-risk rows to an upper quantile; always use the median head. |

#### Feature filtering and domain weighting

| Flag | Type | Default | Valid range / choices | Description |
|------|------|---------|----------------------|-------------|
| `--min-feature-non-null-ratio` | `float` | `0.01` | float in [0, 1) | Drop features with lower non-null rate in train. |
| `--disable-domain-weighting` | flag | off | present / absent | Disable train→test feature reweighting during OOF tuning. |
| `--domain-weight-max` | `float` | `10.0` | float ≥ 1 | Cap on importance weights from the domain classifier. |

#### Intervals

| Flag | Type | Default | Valid range / choices | Description |
|------|------|---------|----------------------|-------------|
| `--interval-alpha` | `float` | `0.1` | float in (0, 1) | Nominal miscoverage for conformal calibration (`0.1` → ~90% target interval). |

**Boolean flags** (`--disable-tail-switch`, `--disable-prior-anchor`, `--disable-domain-weighting`): pass the flag alone to enable; omit it to leave the default (feature enabled).

**Example with several overrides:**

```bash
python train_prism.py \
  --train-file data/train.csv \
  --test-file data/test.csv \
  --target-col my_target \
  --folds 5 \
  --objective log_mae \
  --median-guard-ratio 1.05 \
  --min-rel-gain 0.01 \
  --seed 123
```

---

### 6.2 `measure_error.py`

Used **after** `train_prism.py` when labeled test (or holdout) data are available. Does not affect training.

| Flag | Type | Default | Valid range / choices | Description |
|------|------|---------|----------------------|-------------|
| `--file` | `str` (file path) | `generated.csv` | `.csv`, `.xlsx`, `.xls` | Input table with target and prediction columns. |
| `--target-col` | `str` | see script | column name | Ground-truth column. |
| `--generated-col` | `str` | `generated` | column name | Prediction column to evaluate. |
| `--baseline-col` | `str` | `""` | column name or empty | Optional baseline for side-by-side metrics. Empty or column absent → generated-only table. |
| `--leakage-features` | `str` | see script | comma-separated names | Only used when `--exclude-leakage-rows` is enabled. |
| `--exclude-leakage-rows` | bool | `False` | `--exclude-leakage-rows` / `--no-exclude-leakage-rows` | Drop rows with any non-empty leakage column before metrics. Usually leave **off**. |
| `--trim-worst-fraction` | `float` | `0.05` | float in [0, 0.5) | Fraction of worst rows dropped before **primary** metrics. `0` = all rows. |
| `--trim-by` | `str` | `error` | `error`, `target` | `error`: trim by max(\|gen−target\|, \|base−target\|). `target`: trim by largest \|target\| (stable across runs). |
| `--show-full-metrics` | flag | off | present / absent | Also print metrics on all valid rows (not only the trimmed primary report). |

**Example:**

```bash
python measure_error.py \
  --file generated.csv \
  --target-col esg_firma_esg-bewertung__input__elektrizitaetsverbrauch-kwh \
  --generated-col generated \
  --trim-worst-fraction 0.05 \
  --show-full-metrics
```

---

## 7. How to read the results

After a run, check the console summary and `prism_report.json`:

1. `fallback_to_anchor`
  - `True`: no candidate passed the guards / minimum gain; output equals the anchor (or baseline).  
  - `False`: a model correction was deployed; inspect the chosen `lambda`.
2. **OOF model vs OOF anchor**
  Training cross-validation metrics (MAE, MedianAE, log_mae, …). These support internal review and **are not** a claim about labeled test performance (test labels are not used in training).
3. `oof_stratified` **in** `prism_report.json`
  MAE and win rates by target-magnitude bins—useful to show where the model helps and where the baseline still leads.
4. `generated_trust` **and** `generated_tail_risk`
  Prefer manual review for low-trust or high tail-risk rows.

---



## 8. Quality controls


| Concern           | Measure                                                                                                |
| ----------------- | ------------------------------------------------------------------------------------------------------ |
| Label leakage     | Test target is not used for training, tuning, or interval calibration                                  |
| Prior anchor      | Hierarchical prior is fit out-of-fold so each fold’s labels do not leak into its own prior             |
| Domain shift      | Optional reweighting of training validation rows by feature similarity to the test set (features only) |
| Interval coverage | Conformalized quantile regression (CQR) widens intervals using OOF conformity scores                   |
| Safe deployment   | MedianAE / MAE guards plus anchor fallback                                                             |


---



## 9. Minimal delivery package


| File               | Required                                        |
| ------------------ | ----------------------------------------------- |
| `train_prism.py`   | Yes                                             |
| `METHOD_PRISM.md`  | Recommended                                     |
| `measure_error.py` | Optional (evaluation)                           |
| `requirements.txt` | Recommended (`numpy`, `pandas`, `scikit-learn`) |


Training and test CSVs are provided by the user for each task.

---



## 10. FAQ

**Q: If the test file contains a target column, is it used?**  
A: No. PRISM only reads test features and an optional baseline.

**Q: Can we run without a baseline column?**  
A: Yes. The tool selects categorical fields and builds an empirical-Bayes prior as the anchor.

**Q: Predictions equal the baseline, is that a failure?**  
A: Not necessarily. It usually means fallback was triggered: no candidate passed the guards or the minimum gain. That is intentional safety behavior.

---

*This document matches the current* `train_prism.py` *implementation. Parameter defaults are those in the script’s Defaults section.*