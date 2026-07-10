"""Generate the seven SDLC phase notebooks under notebooks/.

Keeping the notebooks as code (jupytext-style) means they can be regenerated
deterministically and reviewed as plain text in git. Run:  python tools/build_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)

# Every notebook starts by putting src/ on the path so `import car_pricing`
# works whether or not the package was pip-installed.
BOOT = "import sys, warnings\nsys.path.insert(0, '../src')\nwarnings.filterwarnings('ignore')"


def build(name: str, cells: list[tuple[str, str]]) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"}}
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md"
                else nbf.v4.new_code_cell(src) for kind, src in cells]
    path = OUT / name
    nbf.write(nb, path)
    print(f"wrote {path.relative_to(OUT.parent)}  ({len(cells)} cells)")


# ---------------------------------------------------------------------------
# 01 — Business Understanding
# ---------------------------------------------------------------------------
build("01_business_understanding.ipynb", [
    ("md", "# 01 · Business Understanding\n\n"
           "**Phase goal:** frame the problem in *business* terms and define the "
           "measurable KPIs a model must clear before it can ship.\n\n"
           "## The problem\n"
           "A used-car marketplace needs an **instant, trustworthy price estimate** "
           "for any car a seller lists, plus a **budget band** (Low / Medium / High) "
           "for search and filtering. Manual appraisal is slow and inconsistent; a "
           "model that is *fast, accurate and explainable* directly improves listing "
           "conversion and pricing confidence.\n\n"
           "## Stakeholders & value\n"
           "| Stakeholder | What they need | KPI it maps to |\n"
           "| :-- | :-- | :-- |\n"
           "| Seller | A fair, believable price | Low typical error (MAE in ₹) |\n"
           "| Buyer | Sensible budget bands | Band accuracy |\n"
           "| Product | Confidence to auto-suggest | R² / calibration |\n"
           "| Engineering | Something cheap to serve | Small, fast artifact |\n\n"
           "## The ML task\n"
           "A **regression** predicting the exact price (in ₹ Lakhs). The "
           "**Low/Medium/High band is *derived* from the predicted price**, so the "
           "band and the number can never disagree — a deliberate simplification "
           "over training a separate, less-accurate classifier.\n\n"
           "See [`docs/BUSINESS_CASE.md`](../docs/BUSINESS_CASE.md) for the full write-up."),
    ("code", BOOT + "\nfrom car_pricing import config\n"
             "print('KPI gates a model must clear to ship:')\n"
             "for k, v in config.KPI.items():\n    print(f'  {k:20s} {v}')\n"
             "print('\\nPrice unit:', config.PRICE_UNIT, '| Bands:', config.BAND_LABELS)"),
    ("md", "## Success criteria (the ship/no-ship gate)\n\n"
           "The pipeline **automatically refuses to ship** a model that fails any of:\n"
           "- **MAE ≤ ₹1.0 Lakh** — the typical miss must be under ₹1,00,000.\n"
           "- **R² ≥ 0.85** — explain at least 85 % of price variance.\n"
           "- **Band accuracy ≥ 70 %** — correct budget bucket most of the time.\n\n"
           "## SDLC roadmap (the rest of these notebooks)\n"
           "1. Business understanding *(this notebook)*\n"
           "2. Data understanding & EDA\n"
           "3. Data-format benchmarks (CSV vs Parquet vs Feather)\n"
           "4. Feature engineering (target encoding)\n"
           "5. Model comparison (DT vs RF vs XGBoost vs LightGBM …)\n"
           "6. Evaluation & selection\n"
           "7. Productionisation (the shippable pipeline)"),
])

# ---------------------------------------------------------------------------
# 02 — Data Understanding & EDA
# ---------------------------------------------------------------------------
build("02_data_understanding_eda.ipynb", [
    ("md", "# 02 · Data Understanding & EDA\n\n"
           "**Phase goal:** know the data cold — shape, quality, the target's "
           "distribution, and which signals drive price — before modelling."),
    ("code", BOOT + "\nimport pandas as pd, matplotlib.pyplot as plt\n"
             "from car_pricing import config, data\n"
             "raw = data.load_raw()\n"
             "print('shape:', raw.shape)\n"
             "raw.head()"),
    ("code", "print('Missing values per column:')\n"
             "print(raw.isna().sum()[lambda s: s > 0] if raw.isna().any().any() else 'none')\n"
             "print('\\nDtypes:')\nprint(raw.dtypes)"),
    ("md", "### The target: `selling_price` (in Lakhs)\n"
           "Right-skewed, as resale prices usually are — most cars are cheap, a long "
           "tail of premium vehicles. This skew is why we report **MAE** (robust, "
           "intuitive in ₹) alongside R²."),
    ("code", "df = data.clean(raw)\n"
             "print(df[config.TARGET].describe().round(2))\n"
             "ax = df[config.TARGET].clip(upper=40).plot.hist(bins=50, figsize=(8,3))\n"
             "ax.set_xlabel('selling_price (Lakhs)'); ax.set_title('Price distribution'); plt.show()"),
    ("code", "# Cardinality — the modelling challenge that drives feature engineering (nb 04)\n"
             "print('distinct makes :', df['make'].nunique())\n"
             "print('distinct models:', df['model'].nunique())\n"
             "df.groupby('make')[config.TARGET].median().sort_values(ascending=False).head(8)"),
    ("code", "# Numeric drivers of price\n"
             "num = config.NUMERIC + [config.TARGET]\n"
             "df[num].corr()[config.TARGET].sort_values(ascending=False)"),
    ("md", "### Findings that shape the pipeline\n"
           "- **High cardinality:** ~3,200 distinct `model` values → one-hot would "
           "explode the feature space (addressed in nb 04 with target encoding).\n"
           "- **Skewed target:** report MAE in ₹, not just R².\n"
           "- **`max_power` / `engine` correlate strongly with price** → they must be "
           "filled sensibly at serve time (per-model medians), never dropped."),
])

# ---------------------------------------------------------------------------
# 03 — Data Format Benchmarks
# ---------------------------------------------------------------------------
build("03_data_format_benchmarks.ipynb", [
    ("md", "# 03 · Data-Format Benchmarks — CSV vs Parquet vs Feather\n\n"
           "**Phase goal:** choose how to *store* tabular data by measuring the "
           "size ↔ compression ↔ speed ↔ readability trade-off on **this** dataset, "
           "then feed the insight into the production pipeline.\n\n"
           "A CSV is human-readable and opens in Excel, but it's large and stores no "
           "column types. Columnar binary formats (Parquet, Feather) are smaller "
           "and/or far faster and preserve dtypes — at the cost of readability."),
    ("code", BOOT + "\nimport pandas as pd, matplotlib.pyplot as plt\n"
             "from car_pricing import config, data\n"
             "df = data.clean(data.load_raw())\n"
             "res = data.benchmark_formats(df, config.DATA_PROCESSED / '_bench')\n"
             "tbl = (pd.DataFrame(res).T\n"
             "        .assign(KB=lambda t: (t['bytes']/1024).round(0),\n"
             "                vs_csv=lambda t: (t['bytes']/t.loc['csv','bytes']*100).round(1),\n"
             "                write_ms=lambda t: (t['write_s']*1000).round(1),\n"
             "                read_ms=lambda t: (t['read_s']*1000).round(1))\n"
             "        [['KB','vs_csv','write_ms','read_ms']])\n"
             "tbl"),
    ("code", "fig, ax = plt.subplots(1, 2, figsize=(11,3.2))\n"
             "tbl['KB'].plot.barh(ax=ax[0], title='File size (KB) — smaller is better')\n"
             "tbl['read_ms'].plot.barh(ax=ax[1], color='seagreen', title='Read time (ms) — smaller is better')\n"
             "plt.tight_layout(); plt.show()"),
    ("md", "### Takeaways (see [`docs/FORMAT_BENCHMARKS.md`](../docs/FORMAT_BENCHMARKS.md))\n"
           "- **`CSV + gzip`** — ~5× smaller than raw CSV, reads just as fast, pandas "
           "reads it directly. Best *low-friction* shrink for a git repo.\n"
           "- **Parquet** — nearly as small **and ~3–4× faster to read**, keeps dtypes. "
           "The right default for real data pipelines and larger data.\n"
           "- **Feather** — fastest read, but ~3× larger than Parquet; good for "
           "short-lived local hand-offs.\n"
           "- **`CSV + bz2`** — smallest on disk but slowest to read (cold archival only).\n\n"
           "> This project ships `data/raw` as a **gzip-compressed CSV** (`.csv.gz`, ~81% "
           "smaller) — pandas decompresses it transparently on read. The pipeline can still "
           "cache a **Parquet** copy via `data.write_dataframe(...)` for fast reloads."),
])

# ---------------------------------------------------------------------------
# 04 — Feature Engineering
# ---------------------------------------------------------------------------
build("04_feature_engineering.ipynb", [
    ("md", "# 04 · Feature Engineering\n\n"
           "**Phase goal:** turn raw columns into a compact, leakage-safe feature "
           "matrix every model family can consume.\n\n"
           "## The core decision: target-encode `make`/`model`, don't one-hot them\n"
           "With ~3,200 distinct models, one-hot encoding produces a ~3,200-column "
           "matrix — huge, slow, and unusable by `HistGradientBoosting` (no sparse "
           "input). **Target encoding** maps each category to its (cross-fitted) mean "
           "price → just **two** dense numeric columns, no leakage."),
    ("code", BOOT + "\nimport pandas as pd\nfrom car_pricing import config, data, features\n"
             "df = data.clean(data.load_raw())\n"
             "onehot_cols = df['make'].nunique() + df['model'].nunique()\n"
             "print(f'One-hot make+model would add ~{onehot_cols:,} columns')\n"
             "print(f'Target encoding adds exactly {len(config.TARGET_ENCODE)} columns')\n"
             "print(f'Total production features: {len(config.FEATURES)}')"),
    ("code", "# Build the preprocessor and transform a sample to see the compact output\n"
             "X, y = features.split_xy(df)\n"
             "pre = features.build_preprocessor()\n"
             "Xt = pre.fit_transform(X, y)\n"
             "print('encoded matrix shape:', Xt.shape)\n"
             "pd.DataFrame(Xt[:3], columns=config.FEATURES).round(2)"),
    ("code", "# Tercile band edges (in Lakhs) used to DERIVE the band from a price\n"
             "edges = features.band_edges(y)\n"
             "print('band edges:', edges)\n"
             "print('example:', features.price_to_band([2.0, 5.0, 25.0], edges))"),
    ("md", "### Why this is production-grade\n"
           "- **Compact:** 16 features, not ~3,200 → the trained artifact is ~1 MB, "
           "not 100+ MB.\n"
           "- **Leakage-safe:** `TargetEncoder` cross-fits internally, and in CV we "
           "fit it *inside each fold* (see nb 05 / `train.py`).\n"
           "- **Universal:** dense output works for every model, including boosting.\n"
           "- **One contract:** the encoder lives *inside* the sklearn Pipeline, so "
           "the exact same transform is applied at train and serve time."),
])

# ---------------------------------------------------------------------------
# 05 — Model Comparison
# ---------------------------------------------------------------------------
build("05_model_comparison.ipynb", [
    ("md", "# 05 · Model Comparison — the bake-off\n\n"
           "**Phase goal:** fairly compare candidate regressors with cross-validation "
           "and understand *why* the tree ensembles win.\n\n"
           "| Family | How it learns | Expectation |\n"
           "| :-- | :-- | :-- |\n"
           "| **Ridge (linear)** | one weighted sum | underfits — price is non-linear |\n"
           "| **Decision Tree** | one greedy flowchart | overfits, high variance |\n"
           "| **Random Forest** | *bagging* — average of many trees | strong, low-tuning |\n"
           "| **HistGradientBoosting** | *boosting* — trees fix prior errors | strong, fast |\n"
           "| **XGBoost / LightGBM** | regularised boosting | usually best on tabular |"),
    ("code", BOOT + "\nimport json, pandas as pd, matplotlib.pyplot as plt\n"
             "from car_pricing import config\n"
             "# Load the committed comparison (re-run `python -m car_pricing.train` to refresh).\n"
             "comp = json.loads(config.COMPARISON_PATH.read_text())\n"
             "tbl = pd.DataFrame(comp).T.sort_values('cv_mae')\n"
             "tbl[['cv_mae','cv_std','cv_time_s']].round(3)"),
    ("code", "ax = tbl['cv_mae'].plot.barh(xerr=tbl['cv_std'], figsize=(8,3.2), color='steelblue')\n"
             "ax.set_xlabel('CV MAE (Lakhs) — lower is better')\n"
             "ax.set_title('3-fold cross-validated error by model'); ax.invert_yaxis(); plt.show()"),
    ("md", "### DT vs RF vs XGBoost vs LightGBM — what the numbers show\n"
           "- **Linear** is far behind: price depends on non-linear interactions "
           "(brand × age × engine) a single weighted sum can't capture.\n"
           "- A **single tree** improves a lot but is the weakest ensemble baseline.\n"
           "- **Random Forest** (bagging) and the **boosters** cluster at the top; "
           "the gradient boosters (**HistGB / XGBoost / LightGBM**) edge out RF with "
           "fewer, shallower trees — *and* far smaller models.\n"
           "- **XGBoost vs LightGBM** is a near-tie here (well within one std). "
           "Which one *ships* is decided in nb 06 — and it's not purely about MAE.\n\n"
           "See [`docs/MODEL_CARD.md`](../docs/MODEL_CARD.md)."),
])

# ---------------------------------------------------------------------------
# 06 — Evaluation & Selection
# ---------------------------------------------------------------------------
build("06_evaluation_and_selection.ipynb", [
    ("md", "# 06 · Evaluation & Selection\n\n"
           "**Phase goal:** judge the winner on a held-out test set against the "
           "**business KPIs**, and make the final ship decision."),
    ("code", BOOT + "\nimport json\nfrom car_pricing import config\n"
             "m = json.loads(config.METRICS_PATH.read_text())\n"
             "print(f\"CV-best model : {m['cv_best_model']} (MAE {m['cv_best_mae_lakhs']} L)\")\n"
             "print(f\"Shipped model : {m['winner']}\")\n"
             "print(f\"  R^2          : {m['price_r2']}\")\n"
             "print(f\"  MAE          : Rs {m['price_mae_lakhs']} Lakhs\")\n"
             "print(f\"  RMSE         : Rs {m['price_rmse_lakhs']} Lakhs\")\n"
             "print(f\"  Band accuracy: {m['band_accuracy']*100:.1f}%\")\n"
             "print(f\"  KPI gate     : {'PASS' if m['kpi_gate']['all_pass'] else 'FAIL'}\")"),
    ("md", "### The key selection decision: *ship the best model that also deploys*\n"
           "**XGBoost** had the best cross-validated MAE, but the installed "
           "`xgboost 2.1` × `scikit-learn 1.6` combo can't serialise/serve it through "
           "a portable sklearn `Pipeline` (a library-version incompatibility). Since "
           "**LightGBM** is a statistical tie and serves cleanly with no version shim, "
           "`train.py` transparently ships LightGBM. *Operational robustness beats a "
           "0.007-Lakh CV difference* — a real MLOps trade-off, not an accuracy one."),
    ("code", "import pandas as pd, matplotlib.pyplot as plt\n"
             "from sklearn.model_selection import train_test_split\n"
             "from car_pricing import data, features\nimport joblib\n"
             "df = data.clean(data.load_raw()); X, y = features.split_xy(df)\n"
             "_, Xte, _, yte = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)\n"
             "pipe = joblib.load(config.PIPELINE_PATH); pred = pipe.predict(Xte)\n"
             "fig, ax = plt.subplots(1,2, figsize=(11,3.4))\n"
             "ax[0].scatter(yte, pred, s=5, alpha=.3); ax[0].plot([0,40],[0,40],'r--')\n"
             "ax[0].set(xlim=(0,40), ylim=(0,40), xlabel='actual (L)', ylabel='predicted (L)', title='Predicted vs actual')\n"
             "ax[1].hist((pred-yte).clip(-10,10), bins=60, color='indianred'); ax[1].set(title='Residuals (L)')\n"
             "plt.tight_layout(); plt.show()"),
    ("code", "# Band agreement (bands are DERIVED from the predicted price)\n"
             "edges = features.band_edges(y)\n"
             "tb = features.price_to_band(yte, edges); pb = features.price_to_band(pred, edges)\n"
             "pd.crosstab(pd.Series(tb, name='actual'), pd.Series(pb, name='predicted'))"),
    ("md", "### Error analysis\n"
           "Most residuals are tight around zero; the widest misses are premium/rare "
           "models with thin data (a candidate for more data or per-segment models). "
           "The KPI gate passes, so the pipeline is cleared to ship (nb 07)."),
])

# ---------------------------------------------------------------------------
# 07 — Productionisation
# ---------------------------------------------------------------------------
build("07_productionisation.ipynb", [
    ("md", "# 07 · Productionisation\n\n"
           "**Phase goal:** ship the winner as a single, portable artifact with a "
           "clean serving contract — and describe how it deploys.\n\n"
           "## One Pipeline object = no train/serve skew\n"
           "The shipped `.pkl` is a single sklearn `Pipeline` = `TargetEncoder` + model. "
           "The *exact* preprocessing fitted on the training data is re-applied at "
           "inference automatically, so there is zero chance of the encoding drifting "
           "between training and serving."),
    ("code", BOOT + "\nimport joblib, json\nfrom car_pricing import config\n"
             "pipe = joblib.load(config.PIPELINE_PATH)\n"
             "mb = config.PIPELINE_PATH.stat().st_size/1024/1024\n"
             "print(f'artifact: {config.PIPELINE_PATH.name}  ({mb:.2f} MB)')\n"
             "print('steps   :', [s for s,_ in pipe.steps])"),
    ("code", "# The serving contract: give what you know, the rest is auto-filled\n"
             "from car_pricing.predict import predict\n"
             "for car in [{'make':'MARUTI','model':'SWIFT VXI','age':5,'km_driven':40000},\n"
             "            {'make':'BMW','model':'X5','age':4},\n"
             "            {'make':'HYUNDAI','model':'CRETA','fuel':'Diesel','transmission':'Automatic'}]:\n"
             "    print(car, '->', predict(car))"),
    ("code", "meta = json.loads(config.METADATA_PATH.read_text())\n"
             "print('brands       :', len(meta['makes_models']))\n"
             "print('band edges   :', meta['band_edges'])\n"
             "print('numeric dflts:', meta['numeric_defaults'])"),
    ("md", "## How it deploys (see [`docs/PIPELINE_DESIGN.md`](../docs/PIPELINE_DESIGN.md))\n"
           "- **Batch:** `pipe.predict(df)` over a table — nightly repricing.\n"
           "- **REST API:** wrap `car_pricing.predict.predict` in Flask/FastAPI — the "
           "sibling **`app_car_prices_flask`** shows the exact pattern (Docker + ECS).\n"
           "- **Interactive:** a Streamlit UI — the sibling **`app_car_prices_streamlit`**.\n\n"
           "## MLOps loop\n"
           "- **CI:** `pytest` runs the data/feature/KPI-gate contracts on every push.\n"
           "- **CT (continuous training):** re-run `python -m car_pricing.train` on new "
           "data; the KPI gate blocks a regressed model from shipping.\n"
           "- **Monitoring:** track live MAE and input drift; retrain when either "
           "crosses a threshold.\n\n"
           "The whole lifecycle — business KPI → data → features → model bake-off → "
           "evaluation → shippable pipeline — is now reproducible end to end."),
])

# ---------------------------------------------------------------------------
# 08 — XGBoost deep dive (servability & head-to-head) — the "pending" thread
# ---------------------------------------------------------------------------
build("08_xgboost_deep_dive.ipynb", [
    ("md", "# 08 · XGBoost Deep Dive — servability & the head-to-head\n\n"
           "**Phase goal:** close the loop on the model bake-off (nb 05). XGBoost had "
           "the *best* cross-validated MAE, yet the pipeline shipped **LightGBM**. This "
           "notebook answers the two questions that leaves open:\n\n"
           "1. *Why* couldn't XGBoost be shipped through a scikit-learn `Pipeline`?\n"
           "2. If we serve it another way, **is XGBoost actually better** than the "
           "LightGBM we shipped — i.e., did the selection cost us any accuracy?\n\n"
           "Full write-up: [`docs/XGBOOST_SERVABILITY.md`](../docs/XGBOOST_SERVABILITY.md)."),
    ("code", BOOT + "\nimport numpy as np, pandas as pd\n"
             "from sklearn.model_selection import train_test_split\n"
             "from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error\n"
             "from xgboost import XGBRegressor\n"
             "from car_pricing import config, data, features, models\n"
             "from car_pricing.pipeline import make_pipeline\n"
             "df = data.clean(data.load_raw()); X, y = features.split_xy(df)\n"
             "Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)\n"
             "edges = features.band_edges(ytr)\n"
             "print('train/test:', len(Xtr), '/', len(Xte))"),
    ("md", "## 1 · Reproduce the failure\n"
           "XGBoost fits inside a `Pipeline` fine — the break is at **`.predict()`**, "
           "when scikit-learn 1.6 inspects the estimator's tags."),
    ("code", "pipe = make_pipeline(models.model_zoo()['XGBoost'])\n"
             "pipe.fit(Xtr, ytr)          # fitting is fine\n"
             "try:\n"
             "    pipe.predict(Xte[:3])\n"
             "    print('predict OK (env already patched)')\n"
             "except Exception as e:\n"
             "    print('predict FAILS ->', type(e).__name__ + ':', str(e)[:70])"),
    ("md", "## 2 · Root cause — a broken cooperative-inheritance chain\n"
           "scikit-learn 1.6 builds estimator tags by walking the MRO and calling "
           "`super().__sklearn_tags__()` cooperatively. That requires the **mixin first, "
           "`BaseEstimator` last**. XGBoost 2.1.1 declares the bases the other way round, "
           "so `RegressorMixin`'s `super()` lands on `object` — which has no "
           "`__sklearn_tags__` → `AttributeError`."),
    ("code", "mro = [c.__name__ for c in XGBRegressor.__mro__]\n"
             "print('XGBRegressor MRO:', ' -> '.join(mro))\n"
             "print('BaseEstimator before RegressorMixin?',\n"
             "      mro.index('BaseEstimator') < mro.index('RegressorMixin'),\n"
             "      '(that ordering is the bug)')\n"
             "import xgboost, sklearn\n"
             "print('xgboost', xgboost.__version__, '| scikit-learn', sklearn.__version__)"),
    ("md", "### Fix options\n"
           "| Option | What | Verdict |\n"
           "| :-- | :-- | :-- |\n"
           "| **Upgrade** | `pip install 'xgboost>=2.1.2'` (or pin `scikit-learn<1.6`) | ✅ Cleanest — XGBoost then drops straight into the Pipeline |\n"
           "| **Decoupled serving** | Fit the `ColumnTransformer` + `XGBRegressor` separately; `xgb.predict(pre.transform(X))` | ✅ Works on any versions (used below to evaluate) |\n"
           "| Monkey-patch tags | Inject a `__sklearn_tags__` at runtime | ⚠️ Fragile / version-specific — not recommended |"),
    ("md", "## 3 · Serve XGBoost decoupled and evaluate on the held-out test set"),
    ("code", "pre = features.build_preprocessor()\n"
             "Xtr_e = pre.fit_transform(Xtr, ytr); Xte_e = pre.transform(Xte)\n"
             "xgb = XGBRegressor(**models.model_zoo()['XGBoost'].get_params())\n"
             "xgb.fit(Xtr_e, ytr); pred = xgb.predict(Xte_e)\n"
             "def score(yt, p):\n"
             "    tb = features.price_to_band(yt, edges); pb = features.price_to_band(p, edges)\n"
             "    return {'R2': round(r2_score(yt,p),4), 'MAE': round(mean_absolute_error(yt,p),3),\n"
             "            'RMSE': round(float(np.sqrt(mean_squared_error(yt,p))),3),\n"
             "            'band_acc': round(float((tb==pb).mean()),4)}\n"
             "print('XGBoost (decoupled) TEST:', score(yte, pred))"),
    ("md", "## 4 · Head-to-head: XGBoost vs the shipped LightGBM"),
    ("code", "import json\n"
             "comp = json.loads(config.COMPARISON_PATH.read_text())\n"
             "m = json.loads(config.METRICS_PATH.read_text())\n"
             "xgb_test = score(yte, pred)\n"
             "tbl = pd.DataFrame({\n"
             "  'XGBoost':  {'CV MAE': round(comp['XGBoost']['cv_mae'],3),\n"
             "               'Test MAE': xgb_test['MAE'], 'Test R2': xgb_test['R2'],\n"
             "               'Test RMSE': xgb_test['RMSE'], 'Band acc': xgb_test['band_acc'],\n"
             "               'Servable in Pipeline': 'no (this env)'},\n"
             "  'LightGBM (shipped)': {'CV MAE': round(comp['LightGBM']['cv_mae'],3),\n"
             "               'Test MAE': m['price_mae_lakhs'], 'Test R2': m['price_r2'],\n"
             "               'Test RMSE': m['price_rmse_lakhs'], 'Band acc': m['band_accuracy'],\n"
             "               'Servable in Pipeline': 'yes'},\n"
             "})\n"
             "tbl"),
    ("md", "## Conclusion\n"
           "- **The models are statistically indistinguishable on the held-out test "
           "set** — identical R², MAE within ~₹400, band accuracy within 0.2 pt. Note "
           "LightGBM is even *marginally better on test MAE* despite XGBoost's slightly "
           "better CV MAE, confirming the gap is noise.\n"
           "- **Shipping LightGBM therefore cost zero accuracy** while buying clean, "
           "shim-free serialisation and serving — exactly the operational-robustness "
           "trade-off `train.py` automates.\n"
           "- **To ship XGBoost instead**, upgrade to `xgboost>=2.1.2`; it then drops "
           "straight into the same `Pipeline` and the selector would pick it "
           "automatically. Until then, decoupled serving (above) is the escape hatch.\n\n"
           "The bake-off thread is now fully closed."),
])

print("\nAll notebooks generated.")
