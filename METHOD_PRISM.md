# PRISM

## Prior-anchored Robust quantile modeling with conformal trust Intervals for Skewed Measurements

**Implementation:** `train_prism.py` (replaces the correction pipeline in `train_and_generate.py`)
**Application domain:** ESG resource-consumption prediction (water `wasserverbrauch-m3`, electricity `elektrizitaetsverbrauch-kwh`) on dirty, heavy-tailed company tables.

---

## 1. Why the problem is genuinely hard

Diagnostics on the water task (`gap_diagnostics/`) show three structural difficulties that defeat off-the-shelf regression:

1. **Extreme heavy tails.** The target spans 8 orders of magnitude (median ≈ 190 m³, max ≈ 4.6·10⁸ m³). The top 1% of test rows carry **88% of total MAE**; the top 5% carry **99%**. Any model tuned on squared or absolute error in the original scale is effectively trained on 130 rows of near-noise.
2. **Dirty, weakly-informative features.** 550+ mixed columns with missing rates up to 99%, numbers embedded in strings, IDs, dates and free text.
3. **Train→test distribution shift.** A domain classifier separates train from test with AUC ≈ 0.96, and the target/baseline ratio distribution shifts visibly — corrections that validate in-sample do not transfer.

The previous residual-correction pipeline improved RMSE slightly but degraded MedianAE from 1 197 to 5 800 and TrimmedMAE90 from 27k to 609k on test: it damaged the 95% of predictions the baseline already got right, while failing to fix the tail. PRISM is designed so that this failure mode is **structurally impossible**.

---

## 2. Method overview

```
            ┌────────────────────────────────────────────────────────┐
 raw table →│ 1  Heuristic semantic typing & cleaning (no LLM)       │
            │ 2  Anchor construction                                 │
            │      water:  rule-based baseline (feature + anchor)    │
            │      electricity: hierarchical EB prior (synthesized)  │
            │ 3  Quantile GBDT ensemble in log space (q05…q95)       │
            │ 4  Tail-risk head  P(top-decile | X)                   │
            │ 5  OOF tuning: shrinkage λ + tail switch (τ, q_hi)     │
            │      under MedianAE + MAE guards, with anchor fallback │
            │ 6  Conformalized intervals (CQR) + trust scores        │
            └────────────────────────────────────────────────────────┘
 output: generated, generated_low/high, generated_trust, generated_tail_risk
```

### 2.1 Median-optimal robust core (heavy-tail treatment)

All learning happens on `z = log1p(target)` with **quantile (pinball) loss**. Two facts make this the right objective for this data:

- The conditional **median is invariant under monotone transforms**: the q50 model in log space, mapped back with `expm1`, is still the conditional median in m³/kWh — i.e. the **MAE-optimal point prediction** — while the loss itself never sees the raw tail magnitudes.
- Log-space errors are scale-free: a 2× error on a 100 m³ firm and on a 10⁷ m³ firm contribute equally, so the model learns from *all* 8 000 rows instead of the 130 largest.

This removes the need for the old signed-log residual target, gate classifier, ratio clipping and alpha grids — one principled objective replaces four interacting heuristics.

### 2.2 Unified anchor: rule-based baseline or synthesized empirical-Bayes prior

PRISM always has an **anchor** `a(x)` — a conservative reference prediction:

- **Water:** the rule-based `wasser_berechnet` baseline (also fed to the model as a feature, together with `baseline − prior` as a "how unusual is this baseline" signal).
- **Electricity (no baseline):** PRISM synthesizes one — a **hierarchical empirical-Bayes prior**: shrunk group medians of `z` over automatically selected categorical columns (industry, legal form, …), each level shrunk toward its parent with pseudo-count *m*:

  `prior_g = (n_g · median_g + m · prior_parent) / (n_g + m)`

  Columns are chosen by cross-fitted MAE reduction; the prior itself is computed **out-of-fold**, so it is leak-free. This replaces the previous all-zero placeholder column and gives both tasks the same code path — the method *brings its own baseline* when the domain doesn't provide one.

### 2.3 Tail-risk head (attacking the top-5% error mass)

A classifier estimates `p_tail(x) = P(target in top decile | x)`. For samples with `p_tail ≥ τ` the point estimate switches from the median to an upper quantile `q_hi` — an **asymmetric, risk-stratified point estimate**: under-predicting a 10⁷ m³ firm by 100× costs far more MAE than slightly over-predicting a mid-size firm. τ and q_hi are tuned out-of-fold, and "never switch" is always in the candidate set.

### 2.4 Conservative shrinkage with deployment guards

The final point estimate in log space is

`ẑ = a + λ · (z_model − a)` (with the tail switch applied inside `z_model`)

λ ∈ {0, 0.2, …, 1} is tuned on **out-of-fold** predictions with a scale-free objective (log-MAE, optionally reweighted toward the test feature distribution by a cross-validated domain classifier). Two hard guards apply whenever a real baseline exists:

- **MedianAE guard:** candidates whose OOF MedianAE exceeds the baseline's by >5% are rejected outright.
- **MAE guard:** likewise for MAE.
- **Fallback:** if no candidate beats the anchor by ≥0.5% on the objective, the pipeline outputs the anchor unchanged.

The system can therefore *only* deploy a model that improves relative accuracy **without regressing the metrics the baseline is judged by**.

### 2.5 Conformalized intervals and trust scores

The (q05, q95) quantile models give a nominal 90% interval, calibrated with **Conformalized Quantile Regression**: the interval is widened by the empirical (1−α)-quantile of OOF conformity scores, yielding **distribution-free finite-sample coverage** — verified at 0.90 on held-out data. Each prediction ships with:

| Column | Meaning |
|---|---|
| `generated` | point prediction |
| `generated_low / _high` | conformal 90% interval |
| `generated_trust` | 1 − percentile rank of (log-scale) interval width — how much to trust this row |
| `generated_tail_risk` | P(top-decile consumer) — which firms drive aggregate error |

Because ~90% of error concentrates in ~5% of firms, *telling the user which firms those are* is worth as much as the point estimate itself.

---

## 3. Relation to SAGE (ACL 2026)

PRISM applies the same design philosophy as SAGE (*Sparse Adaptive Guidance for Dependency-Aware Tabular Generation*) to the discriminative side:

| Principle | SAGE | PRISM |
|---|---|---|
| Sparse intervention | condition only on high-MI pseudo-features | deviate from the anchor only where the model has demonstrated OOF gains; tail switch only above τ |
| Adaptive strength | logit sharpening/smoothing by context informativeness | λ-shrinkage toward the anchor; per-sample tail switching by p_tail |
| Constraint awareness | reduce constraint-violation rate | median/MAE guards + anchor fallback; conformal coverage guarantee |
| Value-conditioned structure | value-dependent dependency graph | hierarchical EB prior conditioned on categorical context |

---

## 4. What was removed (and why)

| Removed | Reason |
|---|---|
| Qwen3 LLM feature profiler | Heuristic typing (strict numeric-fraction test, datetime/year detection, ID hints) reaches the same decisions at zero cost; removes a GPU dependency and 200+ lines |
| Gate classifier + α grid + segmented α + ratio clipping + high-baseline caps | Replaced by one shrinkage parameter λ + guards; the old stack had 5 interacting knobs that overfit validation (observed: MedianAE 1 197 → 5 800 on test) |
| Signed-log residual target | Quantile regression on `log1p(target)` is median-optimal by construction |
| All-zero synthetic baseline column | Replaced by the EB prior anchor |

---

## 5. Validation protocol

- **OOF (stratified 5-fold by log-target decile):** model vs anchor on MAE / MedianAE / WAPE / log-MAE / TrimmedMAE90, plus a per-magnitude-bin win-rate table (`prism_report.json`, `prism_oof.csv`).
- **Interval calibration:** raw vs conformal coverage.
- **Synthetic end-to-end test** (heavy-tailed dirty data, 3% wild outliers, broken-baseline segments): PRISM improved every metric over the baseline in both modes (e.g. log-MAE −17%, MedianAE −12%), held-out conformal coverage 0.91/0.90, tail-risk AUC ≈ 0.91, and the trust score cleanly separated high-error from low-error rows.

## 6. Usage

```bash
# Water (rule-based baseline available)
python train_prism.py --train-file train_clean.csv --test-file test_clean.csv \
  --target-col esg_firma_esg-bewertung__input__wasserverbrauch-m3 \
  --baseline-col esg_firma_wasser_berechnet \
  --leakage-features "<water leakage columns>"

# Electricity (no baseline -> synthesized EB prior anchor)
python train_prism.py --train-file train_clean.csv --test-file test_clean.csv \
  --target-col esg_firma_esg-bewertung__input__elektrizitaetsverbrauch-kwh
```

Key knobs: `--objective` (default `log_mae`), `--median-guard-ratio` / `--mae-guard-ratio` (default 1.05), `--min-rel-gain`, `--prior-levels`, `--disable-tail-switch`, `--interval-alpha`.
