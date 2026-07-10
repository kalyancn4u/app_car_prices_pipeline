# 🃏 Model Card — Used-Car Price Predictor

## Overview

| | |
| :-- | :-- |
| **Task** | Regression → price in ₹ Lakhs (band derived from price) |
| **Shipped model** | **LightGBM** regressor inside a scikit-learn `Pipeline` |
| **Preprocessing** | `TargetEncoder(make, model)` + passthrough numeric/flags → **16 features** |
| **Training data** | 15,856 rows (80 % of 19,820); held-out test 3,964 |
| **Artifact** | `models/price_pipeline.pkl` — **0.92 MB** (LZMA) |

## The bake-off (3-fold CV, MAE in Lakhs — lower is better)

Reproducible in [`notebooks/05_model_comparison.ipynb`](../notebooks/05_model_comparison.ipynb);
preprocessing is fit **inside each fold** (no target leakage).

| Model | How it learns | CV MAE | ±std | CV time |
| :---- | :------------ | -----: | ---: | ------: |
| Ridge (linear) | one weighted sum | 1.184 | 0.027 | 0.1 s |
| Decision Tree | one greedy flowchart | 0.860 | 0.005 | 0.2 s |
| Random Forest | bagging (avg. of many trees) | 0.745 | 0.004 | 2.8 s |
| HistGradientBoosting | boosting (trees fix prior errors) | 0.732 | 0.006 | 2.1 s |
| **XGBoost** | regularised boosting | **0.708** | 0.007 | 1.5 s |
| **LightGBM** | leaf-wise boosting | **0.715** | 0.004 | 3.0 s |

### DT vs RF vs XGBoost vs LightGBM — reading the results

- **Linear** trails badly: price is driven by non-linear interactions
  (brand × age × engine) a single weighted sum can't express.
- A **single Decision Tree** improves greatly but is the weakest ensemble baseline
  (high variance).
- **Random Forest** (bagging) and the **gradient boosters** cluster at the top; the
  boosters (**HistGB / XGBoost / LightGBM**) edge out RF with fewer, shallower trees
  **and** far smaller models.
- **XGBoost vs LightGBM** is a **statistical tie** here (0.708 vs 0.715, within one
  std of each other).

## Selection decision: *ship the best model that also deploys*

XGBoost had the best CV MAE, but the installed **`xgboost 2.1` × `scikit-learn 1.6`**
combination cannot serialise/serve XGBoost through a portable sklearn `Pipeline`
(a known library-version incompatibility in the tag system). Because LightGBM is a
statistical tie and serves cleanly with **no version shim**, the pipeline
transparently ships **LightGBM**. This is a genuine MLOps trade-off — *operational
robustness beats a 0.007-Lakh CV difference* — and it is automated in `train.py`
(it walks the CV ranking and ships the first model that fits **and** predicts in a
Pipeline).

**Was XGBoost actually better?** No — evaluated via decoupled serving on the same
held-out test set, the two are indistinguishable, and LightGBM is even marginally
better on test MAE:

| Metric | XGBoost | LightGBM (shipped) |
| :----- | ------: | -----------------: |
| CV MAE | 0.708 | 0.715 |
| Test MAE | 0.668 | **0.664** |
| Test R² | 0.9568 | 0.9568 |
| Band accuracy | 0.8605 | 0.8582 |

So shipping LightGBM cost **zero accuracy**. Full root-cause analysis, fix options
and reproduction: [`XGBOOST_SERVABILITY.md`](XGBOOST_SERVABILITY.md) /
[notebook 08](../notebooks/08_xgboost_deep_dive.ipynb). To ship XGBoost instead,
upgrade to `xgboost>=2.1.2` and re-run training — no code changes needed.

## Held-out test performance (shipped LightGBM)

| Metric | Value | KPI gate |
| :----- | :---- | :------- |
| R² | **0.957** | ≥ 0.85 ✅ |
| MAE | **₹0.66 Lakhs** (≈ ₹66,000) | ≤ ₹1.0 L ✅ |
| RMSE | ₹1.02 Lakhs | — |
| Band accuracy (derived) | **85.8 %** | ≥ 70 % ✅ |

## The band is derived, not classified

The Low/Medium/High band comes from mapping the **predicted price** onto the
training terciles **[0.3, 3.99, 6.75, 20.9] L**. Guarantees: (1) the band and the
₹ figure never contradict each other; (2) no second model to train, tune or drift.
At **85.8 %** it also beats the separate-classifier approach used by the sibling
projects (~76.9 %).

## How this compares to the sibling projects

| | This (MLOps) | Streamlit / Flask siblings |
| :-- | :-- | :-- |
| Encoding | Target encoding (2 cols) | One-hot make+model (~3,200 cols) |
| Features | 16 | ~3,250 |
| Artifact | **0.92 MB** | 8.9–13.5 MB |
| Price R² | **0.957** | 0.950 |
| Band accuracy | **85.8 %** (derived) | 76.9 % (classifier) |

Same data, better numbers, ~15× smaller model — the payoff of the feature-encoding
and band-derivation decisions.

## Limitations & ethical notes

- No condition/accident/service-history features → systematic misses on unusually
  well- or poorly-kept cars.
- Rare/premium models have thin data → wider errors (visible in the residuals in
  [`notebooks/06_evaluation_and_selection.ipynb`](../notebooks/06_evaluation_and_selection.ipynb)).
- Predictions are decision support, **not** a guaranteed sale price.
- Retraining refreshes the terciles and target encodings; monitor drift.
