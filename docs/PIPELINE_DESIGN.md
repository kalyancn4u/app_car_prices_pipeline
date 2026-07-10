# 🏗️ Pipeline Design

How the code is organised, why, and how it deploys — the "production-grade"
reasoning behind the package.

## One `Pipeline` object = no train/serve skew

The single most important decision: preprocessing and the model are bundled into
**one** scikit-learn `Pipeline` and serialised together.

```
price_pipeline.pkl
 └── Pipeline
      ├── preprocess : ColumnTransformer
      │     ├── TargetEncoder(["make","model"])      # 2 dense columns
      │     └── passthrough(numeric + flags)          # 14 columns
      └── model      : LightGBMRegressor
```

Because the *fitted* `TargetEncoder` travels inside the artifact, the exact
encoding learned at training time is re-applied at inference automatically. There
is no separate "feature script" that could drift out of sync — the classic cause
of train/serve skew. Ship one file, call `.predict()`.

## Package layout (`src/car_pricing/`)

| Module | Responsibility |
| :----- | :------------- |
| `config.py` | Single source of truth: paths, schema, split, **KPI gates**. |
| `data.py` | Load/clean; **format-agnostic I/O** (CSV/Parquet/Feather) + the format benchmark. |
| `features.py` | The preprocessor (target encoding), band edges, `price_to_band`, serving metadata. |
| `models.py` | The candidate **model zoo** (Ridge, DT, RF, HistGB, XGBoost, LightGBM). |
| `pipeline.py` | `make_pipeline(estimator)` — preprocessor + model. |
| `train.py` | Orchestrates: CV bake-off → select servable winner → evaluate → **KPI gate** → save. |
| `predict.py` | Serving surface: friendly payload → price + derived band. |

Notebooks and tests **import this package** rather than re-implementing logic, so
there is exactly one definition of every step.

> **"Servable winner" in practice:** `train.py` walks the CV ranking and ships the
> first model that both fits **and** predicts inside a portable `Pipeline`. On this
> environment that skips XGBoost (a `xgboost 2.1 × sklearn 1.6` tag bug) in favour
> of LightGBM — at zero accuracy cost. Root cause, fix options and the head-to-head
> are in [`XGBOOST_SERVABILITY.md`](XGBOOST_SERVABILITY.md).

## Why target encoding (not one-hot)

`model` has 3,233 levels. One-hot → a ~3,200-column matrix that is huge, slow, and
unusable by `HistGradientBoosting` (dense-only). `TargetEncoder` maps each category
to its **cross-fitted mean price** → two dense columns, no leakage, usable by every
model family. Result: **16 features**, a **0.92 MB** artifact (vs 8.9–13.5 MB for
the one-hot siblings), and *higher* accuracy. See
[MODEL_CARD.md](MODEL_CARD.md).

## Leakage control

- **In CV:** the encoder is fit **inside each fold** (fit on the fold's train rows,
  transform its validation rows) — see `train.cross_validate_zoo`.
- **In the final fit:** `TargetEncoder` cross-fits internally on the training set.
- The **held-out test set** never participates in model selection or encoding.

## The KPI gate (automated ship/no-ship)

`train.py` computes R² / MAE / band accuracy on the held-out test set and checks
them against `config.KPI`. The result is written to `models/metrics.json` as
`kpi_gate.all_pass`; a CI job (or a human) can block deployment on it. This turns
"is the model good enough?" from a judgement call into a **reproducible gate**.

## Deployment options

The same artifact serves three shapes (the two siblings implement the first two):

- **REST API** — wrap `car_pricing.predict.predict` in Flask/FastAPI + Docker + ECS.
  See the sibling **`app_car_prices_flask`** for the exact pattern.
- **Interactive UI** — a Streamlit front-end. See **`app_car_prices_streamlit`**.
- **Batch** — `pipeline.predict(df)` over a table for nightly repricing.

## MLOps loop

| Stage | This repo |
| :---- | :-------- |
| **CI** | `pytest` runs data/feature/KPI-gate contracts on every push. |
| **CT** (continuous training) | Re-run `python -m car_pricing.train` on new data; the KPI gate blocks regressions. |
| **CD** | Commit the refreshed `models/` artifacts (they're small) or publish them as a build artifact. |
| **Monitoring** | Track live MAE vs the ₹1 L gate and input drift; either crossing a threshold triggers a retrain. |

## Reproducibility

A fixed `random_state=42` governs the split, the CV folds, and every estimator, so
`python -m car_pricing.train` reproduces the reported metrics bit-for-bit on the
same data and library versions.
