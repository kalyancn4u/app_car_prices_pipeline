# 🧑‍🎓 Beginner's Guide

Welcome! This guide explains the whole project in plain English, with **no assumed
background**. If you can follow a recipe, you can follow this. By the end you'll be able to
run the pipeline, make a prediction, and start extending it yourself.

---

## 1. What is this project?

It predicts the **resale price of a used car** (in Indian ₹ Lakhs) from a few facts about the
car — its make, model, age, and kilometres driven. It's not just a model; it's the **whole
pipeline** around one: preparing data, comparing models, evaluating the winner, and packaging
it so an app could serve it.

Think of it as a small, honest example of *how a real ML project is built end to end*.

> **Lakh?** An Indian unit: 1 Lakh = ₹100,000. So "₹6.50 Lakhs" ≈ ₹650,000.

---

## 2. The 5-minute setup

You need **Python 3.9+** and this repository on your computer.

```bash
# 1. Create a private sandbox for this project's libraries
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # macOS / Linux

# 2. Install what it needs
pip install -r requirements.txt
pip install -e .                 # lets you write `import car_pricing`

# 3. Make a prediction (the trained model already ships with the repo)
python -c "from car_pricing.predict import predict; print(predict({'make':'MARUTI','model':'SWIFT VXI','age':5,'km_driven':40000}))"
```

You should see something like:

```python
{'predicted_price_lakhs': 5.74, 'predicted_price_display': '₹5.74 Lakhs', 'price_band': 'Medium'}
```

That's it — you just ran a machine-learning model. 🎉

> **Why does it work without training?** The trained model (`models/price_pipeline.pkl`) and
> the dataset are committed to the repo, so a fresh copy runs immediately. To rebuild the
> model yourself, run `python -m car_pricing.train`.

---

## 3. What's in the folders?

| Folder / file | In plain words |
| :------------ | :------------- |
| `data/raw/…csv.gz` | The dataset — 19,820 real car listings, stored **gzip-compressed** to save space (pandas reads it directly). |
| `src/car_pricing/` | The **code** of the pipeline, split into small files (load data, make features, train, predict). |
| `notebooks/` | Eight **Jupyter notebooks**, one per project phase — read them in order (01 → 08) to see the whole story with charts. |
| `models/` | The **trained model** and its scorecards (`metrics.json`), saved to disk. |
| `docs/` | Written explanations (you're reading one). |
| `tests/` | Automated checks. `test_pipeline.py` already works; `test_stubs.py` is **your** to-do list. |
| `tools/` | A helper that regenerates the notebooks. |

---

## 4. A few words you'll meet (mini-glossary)

| Term | Plain meaning |
| :--- | :------------ |
| **Model** | A "brain" that learned patterns from past data and can now guess new answers. |
| **Regression** | Predicting a **number** (here, the exact price). |
| **Classification** | Predicting a **category** (here, a Low / Medium / High budget band). |
| **Feature** | One input column the model looks at (age, km, engine size…). |
| **Target encoding** | A trick to turn a text column with thousands of values (like *model*) into one useful number, instead of thousands of 0/1 columns. |
| **Pipeline** | Preprocessing + model bundled into **one object**, so training and serving can't drift apart. |
| **Cross-validation** | Testing a model on data it *didn't* train on, several times, to get an honest score. |
| **KPI gate** | A pass/fail check: the model only "ships" if it clears the business targets (error small enough, etc.). |
| **Drift** | When new real-world data slowly stops looking like the training data — a signal to retrain. |

---

## 5. The project, phase by phase (the notebooks)

Each notebook is a chapter of the story. Open them in order:

| # | Notebook | What you'll learn |
| :- | :------- | :---------------- |
| 01 | Business understanding | *Why* we're building this and how we'll judge success (the KPIs). |
| 02 | Data understanding (EDA) | Getting to know the data: distributions, missing values, what drives price. |
| 03 | Data-format benchmarks | Why we store the data as `.csv.gz`; CSV vs Parquet vs Feather, measured. |
| 04 | Feature engineering | Turning raw columns into model-ready features (the target-encoding trick). |
| 05 | Model comparison | A fair bake-off: Decision Tree vs Random Forest vs XGBoost vs LightGBM. |
| 06 | Evaluation & selection | Judging the winner on unseen data against the KPIs. |
| 07 | Productionisation | Packaging the winner as one shippable file. |
| 08 | XGBoost deep dive | A real-world tooling snag and how it was investigated and resolved. |

> **Tip:** you can just *read* the notebooks on GitHub — they already contain their outputs
> and charts. To run them yourself: `pip install jupyter && jupyter lab notebooks/`.

---

## 6. How a prediction actually flows

```
 you: {"make":"MARUTI","model":"SWIFT VXI","age":5,"km_driven":40000}
        │
        ▼   (car_pricing.predict.predict)
 fill in the specs the model needs (engine, power…) from this model's typical values
        │
        ▼   (the saved Pipeline: target-encode → LightGBM)
 predicted price  ── mapped onto price bands ──►  "₹5.74 Lakhs · Medium"
```

Everything lives in small, readable functions under `src/car_pricing/` — open `predict.py`
and `features.py` to see exactly these steps.

---

## 7. Your turn — learn by completing the test stubs

The file [`tests/test_stubs.py`](../tests/test_stubs.py) is a **guided to-do list**. Each
entry is a named test that is *skipped* and has a docstring telling you what to check.

```bash
pytest -v          # working tests pass; stubs show as SKIPPED
```

To complete one:
1. Pick a stub (start with the **Data** group — the easiest).
2. Delete its `@pytest.mark.skip(...)` line.
3. Write the check described in its docstring (copy patterns from `tests/test_pipeline.py`).
4. Re-run `pytest` until it's green. ✅

They climb from pandas basics up to the "pending" MLOps stages (hyper-parameter tuning,
drift detection) — a ready-made learning path.

---

## 8. Where to go next

- 💼 [`BUSINESS_CASE.md`](BUSINESS_CASE.md) — the problem and the success criteria.
- 🃏 [`MODEL_CARD.md`](MODEL_CARD.md) — how the models were compared and chosen.
- 🏗️ [`PIPELINE_DESIGN.md`](PIPELINE_DESIGN.md) — how the code is organised and why.
- 🗃️ [`FORMAT_BENCHMARKS.md`](FORMAT_BENCHMARKS.md) — the CSV/Parquet/Feather comparison.

Take it one notebook, one stub at a time. You've got this. 🚀
