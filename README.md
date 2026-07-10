# 🚗🛠️ Car Price Pipeline — a beginner-friendly production-ML starter

> A hands-on, **end-to-end machine-learning pipeline** for used-car price prediction —
> from business KPIs to a deployable, KPI-gated model — packaged as a **learning starter**
> you can read, run, and extend.

> 🧑‍🎓 **New here? Start with the [Beginner's Guide](docs/BEGINNERS_GUIDE.md)** — a
> plain-English tour of the whole project that assumes no prior background.

This project is a **learning-focused variant** of
[`app_mlops_car_prices`](https://github.com/kalyancn4u/app_mlops_car_prices): the same
working pipeline, plus a gentle guide and a scaffold of **test stubs** for you to complete
as you learn. It is one of the **Car Prices Quartet**, four repositories on the same Cars24 dataset:

| Repo | What it is |
| :--- | :--------- |
| `app_streamlit_car_prices` | An interactive Streamlit app |
| `app_flask_car_prices` | A containerised Flask REST API (Docker + AWS ECS) |
| `app_mlops_car_prices` | The full ML lifecycle, done properly and reproducibly |
| **`app_pipeline_car_prices`** *(this repo)* | A **beginner-friendly starter** — the pipeline + guided docs + **test stubs to extend** |

> **What's different here:** the working pipeline is unchanged from the MLOps sibling; what's
> added is a [Beginner's Guide](docs/BEGINNERS_GUIDE.md) and a set of guided test stubs
> (`tests/test_stubs.py`) so you can learn by filling them in — see
> [Your turn: extend the project](#-your-turn-extend-the-project).

---

## 📑 Contents

1. [Headline results](#-headline-results)
2. [The lifecycle (notebooks)](#-the-lifecycle-notebooks)
3. [Quick start](#-quick-start)
4. [How the pipeline works](#-how-the-pipeline-works)
5. [Model comparison (DT vs RF vs XGBoost vs LightGBM)](#-model-comparison-dt-vs-rf-vs-xgboost-vs-lightgbm)
6. [Data formats: CSV vs Parquet vs Feather](#-data-formats-csv-vs-parquet-vs-feather)
7. [Project structure](#-project-structure)
8. [How it compares to the sibling projects](#-how-it-compares-to-the-sibling-projects)
9. [Your turn: extend the project](#-your-turn-extend-the-project)
10. [Documentation index](#-documentation-index)

---

## 🎯 Headline results

Trained on 19,820 listings (41 makes, 3,233 models), held-out 20 % test set:

| Metric | Result | KPI gate |
| :----- | :----- | :------- |
| **Price R²** | **0.957** | ≥ 0.85 ✅ |
| **Price MAE** | **₹0.66 Lakhs** (≈ ₹66,000) | ≤ ₹1.0 L ✅ |
| **Band accuracy** (derived) | **85.8 %** | ≥ 70 % ✅ |
| **Shipped artifact** | **0.92 MB** | — |

Shipped model: **LightGBM** in a scikit-learn `Pipeline`. The pipeline **refuses to
ship** a model that fails any KPI gate — "good enough?" is a reproducible check, not
a judgement call.

---

## 🧭 The lifecycle (notebooks)

Every SDLC phase is a runnable, **already-executed** notebook (outputs + charts
embedded). They import the `car_pricing` package rather than duplicating logic.

| # | Notebook | Phase | You'll see |
| :- | :------- | :---- | :--------- |
| 01 | [`01_business_understanding`](notebooks/01_business_understanding.ipynb) | Business KPIs | Problem framing, stakeholders, the ship/no-ship gate |
| 02 | [`02_data_understanding_eda`](notebooks/02_data_understanding_eda.ipynb) | Data & EDA | Distributions, missingness, cardinality, price drivers |
| 03 | [`03_data_format_benchmarks`](notebooks/03_data_format_benchmarks.ipynb) | Storage | Measured CSV vs Parquet vs Feather (size/speed) |
| 04 | [`04_feature_engineering`](notebooks/04_feature_engineering.ipynb) | Features | Target encoding vs one-hot; band edges |
| 05 | [`05_model_comparison`](notebooks/05_model_comparison.ipynb) | Modelling | The DT/RF/XGB/LGBM bake-off, cross-validated |
| 06 | [`06_evaluation_and_selection`](notebooks/06_evaluation_and_selection.ipynb) | Evaluation | Held-out metrics, residuals, KPI gate, ship decision |
| 07 | [`07_productionisation`](notebooks/07_productionisation.ipynb) | Deployment | The one-Pipeline artifact + serving contract |
| 08 | [`08_xgboost_deep_dive`](notebooks/08_xgboost_deep_dive.ipynb) | Appendix | Why XGBoost couldn't ship, and the XGBoost-vs-LightGBM head-to-head |

---

## ⚡ Quick start

```bash
# 1. Environment
python -m venv venv && venv\Scripts\Activate.ps1     # Windows PowerShell
pip install -r requirements.txt
pip install -e .                                      # makes `car_pricing` importable

# 2. Reproduce the whole model pipeline (bake-off -> select -> evaluate -> save)
python -m car_pricing.train

# 3. Predict
python -c "from car_pricing.predict import predict; print(predict({'make':'MARUTI','model':'SWIFT VXI','age':5,'km_driven':40000}))"

# 4. Explore the lifecycle
jupyter lab notebooks/      # 01 -> 07

# 5. Tests
pytest -q
```

> The trained artifacts are **committed** (they're ~1 MB total), so steps 3–5 work
> immediately on a fresh clone; step 2 only re-creates them.

Example output:

```python
{'predicted_price_lakhs': 5.74, 'predicted_price_display': '₹5.74 Lakhs', 'price_band': 'Medium'}
```

---

## 🔧 How the pipeline works

```
   data/raw/*.csv
        │  clean
        ▼
   ┌─────────────────────────── one sklearn Pipeline ───────────────────────────┐
   │  ColumnTransformer                                     Regressor            │
   │   ├─ TargetEncoder(make, model)   → 2 dense cols   ┐                        │
   │   └─ passthrough(numeric + flags) → 14 cols        ├─►  LightGBM  → ₹ price │
   └────────────────────────────────────────────────────┘         │            │
                                                                   ▼            │
                                          price → tercile edges → Low/Med/High band
```

The **whole thing serialises to one 0.92 MB `.pkl`** — the fitted target-encoder
rides along with the model, so training and serving can't drift apart. Full
rationale in [`docs/PIPELINE_DESIGN.md`](docs/PIPELINE_DESIGN.md).

**Key production decisions:**

- **Target-encode `make`/`model`** instead of one-hot → 16 features (not ~3,200), a
  ~15× smaller model, *and* higher accuracy.
- **Derive the band from the predicted price** → the band and the ₹ figure can never
  disagree, and there's no second model to train or drift.
- **KPI-gated selection** → the pipeline ships the **best model that also deploys**
  cleanly (see the model comparison below).

---

## 📊 Model comparison (DT vs RF vs XGBoost vs LightGBM)

3-fold cross-validated MAE (Lakhs), preprocessing fit inside each fold — full table
and discussion in [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) and
[notebook 05](notebooks/05_model_comparison.ipynb):

| Model | CV MAE | Note |
| :---- | -----: | :--- |
| Ridge (linear) | 1.184 | underfits — price is non-linear |
| Decision Tree | 0.860 | high variance |
| Random Forest | 0.745 | strong, low-tuning |
| HistGradientBoosting | 0.732 | fast boosting |
| **XGBoost** | **0.708** | best CV — but not servable in a Pipeline here* |
| **LightGBM** | **0.715** | statistical tie → **shipped** (deploys cleanly) |

\* The installed `xgboost 2.1 × scikit-learn 1.6` can't serialise/serve XGBoost
through a portable Pipeline. Since LightGBM is a tie and serves with no version
shim, `train.py` ships it automatically — **operational robustness over a
0.007-Lakh difference**. That's a real MLOps trade-off, documented, not hidden.
Evaluated head-to-head on the test set, XGBoost is **no better** (LightGBM is even
marginally ahead on MAE) → **zero accuracy cost**. Full analysis:
[`docs/XGBOOST_SERVABILITY.md`](docs/XGBOOST_SERVABILITY.md) ·
[notebook 08](notebooks/08_xgboost_deep_dive.ipynb).

---

## 🗃️ Data formats: CSV vs Parquet vs Feather

Measured on this dataset (full table + the "why does it open in Excel?" explainer
in [`docs/FORMAT_BENCHMARKS.md`](docs/FORMAT_BENCHMARKS.md) and
[notebook 03](notebooks/03_data_format_benchmarks.ipynb)):

| Format | Size | Read | Best for |
| :----- | ---: | ---: | :------- |
| CSV (raw) | 1,535 KB | 23 ms | Humans, Excel, tiny files |
| CSV + gzip | 290 KB | 26 ms | Low-friction repo shrink (pandas reads it directly) |
| Parquet (zstd) | 227 KB | **8 ms** | Real pipelines & large data (columnar, keeps dtypes) |
| Feather | 867 KB | **5 ms** | Fastest read, temporary local hand-offs |

`data/raw/` stays a plain CSV for transparency; the pipeline can cache a Parquet
copy for fast, type-safe reloads.

---

## 📁 Project structure

```
app_pipeline_car_prices/
├── README.md
├── requirements.txt · pyproject.toml · Makefile · .gitignore
├── data/
│   ├── raw/cars24-car-price-cleaned-new.csv.gz # the dataset, gzip-compressed (committed)
│   └── processed/                               # generated (gitignored)
├── notebooks/                # 01–08, executed with outputs & charts
├── src/car_pricing/          # the production package
│   ├── config.py · data.py · features.py · models.py
│   ├── pipeline.py · train.py · predict.py
├── models/                   # trained artifacts (committed, ~1 MB)
│   ├── price_pipeline.pkl · metrics.json · model_comparison.json · serving_metadata.json
├── docs/                     # BEGINNERS_GUIDE · BUSINESS_CASE · DATA_DICTIONARY · MODEL_CARD
│   │                         # PIPELINE_DESIGN · FORMAT_BENCHMARKS · XGBOOST_SERVABILITY
├── tests/
│   ├── test_pipeline.py      # working data/feature/KPI-gate/serving contracts
│   └── test_stubs.py         # guided TODO stubs for you to complete
└── tools/build_notebooks.py  # regenerates the notebooks deterministically
```

---

## 🔗 How it compares to the sibling projects

Same data, a deliberately different (production) approach:

| | This pipeline | Streamlit / Flask siblings |
| :-- | :-- | :-- |
| Encoding | Target encoding (2 cols) | One-hot make+model (~3,200 cols) |
| Model artifact | **0.92 MB** | 8.9–13.5 MB |
| Price R² | **0.957** | 0.950 |
| Band | **Derived** from price (85.8 %) | Separate classifier (76.9 %) |
| Focus | The **lifecycle** & pipeline | Serving the model to users |

The siblings answer *"how do users interact with the model?"*; this repo answers
*"how do you build, compare, and ship the model responsibly?"*

---

## 🧑‍🎓 Your turn: extend the project

This starter is meant to be *finished by you*. Two on-ramps:

1. **Read the [Beginner's Guide](docs/BEGINNERS_GUIDE.md)** — it explains every part in plain
   English and shows how to run the pipeline end to end.
2. **Complete the test stubs** in [`tests/test_stubs.py`](tests/test_stubs.py). Each is a
   named, *skipped* test with a docstring describing what to check:

   ```bash
   pytest -v        # the real tests pass; the stubs show as SKIPPED
   ```

   Remove a stub's `@pytest.mark.skip(...)` marker and implement the body to turn it green.
   They are ordered from easy to advanced:

   | Group | Example stub | Skill it builds |
   | :---- | :----------- | :-------------- |
   | Data | cleaning drops invalid rows | pandas basics |
   | Features | target encoding adds no leakage | preprocessing |
   | Serving | an unknown make/model is rejected | error handling |
   | Modelling | a tuned model beats the baseline | hyper-parameter search |
   | Monitoring | drift is flagged on shifted data | the "pending" MLOps stages |

Each stub maps to a natural extension — hyper-parameter tuning (Optuna), experiment tracking
(MLflow), drift detection (Evidently), a REST/Streamlit serving layer, or a CI workflow.
[`docs/PIPELINE_DESIGN.md`](docs/PIPELINE_DESIGN.md) shows where each fits.

---

## 📚 Documentation index

- 🧑‍🎓 [`docs/BEGINNERS_GUIDE.md`](docs/BEGINNERS_GUIDE.md) — plain-English tour of the whole project (**start here**)
- 💼 [`docs/BUSINESS_CASE.md`](docs/BUSINESS_CASE.md) — problem, value, KPIs, scope
- 📖 [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) — every column, the baselines, cleaning
- 🃏 [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) — the bake-off, selection, performance, limits
- 🧩 [`docs/XGBOOST_SERVABILITY.md`](docs/XGBOOST_SERVABILITY.md) — why XGBoost didn't ship, and the head-to-head
- 🏗️ [`docs/PIPELINE_DESIGN.md`](docs/PIPELINE_DESIGN.md) — package layout, leakage control, MLOps loop
- 🗃️ [`docs/FORMAT_BENCHMARKS.md`](docs/FORMAT_BENCHMARKS.md) — CSV vs Parquet vs Feather, the Excel trap

---

> ⚠️ **Disclaimer:** predictions are statistical estimates from historical Cars24
> listings — decision support, not a guaranteed sale price.

<sub>Built with scikit-learn · LightGBM · XGBoost · pandas · pyarrow. A learning-focused
member of the `*_car_prices` quartet.</sub>

---

### 🔗 The Car Prices Quartet

Four sibling projects built on the same Cars24 dataset:

- 🎛️ **[Streamlit web app →](https://github.com/kalyancn4u/app_streamlit_car_prices)** — interactive price-predictor UI
- 🐳 **[Flask REST API →](https://github.com/kalyancn4u/app_flask_car_prices)** — containerised API (Docker + AWS ECS/Fargate)
- 🔬 **[MLOps lifecycle →](https://github.com/kalyancn4u/app_mlops_car_prices)** — full SDLC: notebooks → production pipeline
- 🛠️ **Pipeline starter** — beginner-friendly guide + test stubs to extend · _you are here_
