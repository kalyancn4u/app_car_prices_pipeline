# 📗 API Examples — the `car_pricing` package by example

Copy-paste recipes for the pipeline package. Each shows the **import**, the **call**,
and the **expected output**. Runnable version:
[`notebooks/api_examples.ipynb`](../notebooks/api_examples.ipynb). New to the project?
Read the [Beginner's Guide](BEGINNERS_GUIDE.md) first.

Setup once:

```bash
pip install -r requirements.txt && pip install -e .
```

---

## Predict a price — `predict()`

```python
from car_pricing.predict import predict

predict({"make": "MARUTI", "model": "SWIFT VXI", "age": 5, "km_driven": 40000})
# -> {'predicted_price_lakhs': 5.74,
#     'predicted_price_display': '₹5.74 Lakhs',
#     'price_band': 'Medium'}
```

Only `make` + `model` are required — the rest is auto-filled from that model's typical
values:

```python
predict({"make": "BMW", "model": "X5"})     # -> ~₹18–20 Lakhs, band 'High'
```

## Load & clean the data

```python
from car_pricing import data

df = data.clean(data.load_raw())            # 19,820 rows, cleaned
df.shape                                    # (19820, 18)

# Format-agnostic I/O (pandas infers the type from the extension):
data.write_dataframe(df.head(100), "sample.parquet")
data.read_dataframe("sample.parquet").shape
```

## Features — encoding & bands

```python
from car_pricing import features

X, y = features.split_xy(df)                 # X: the 16 model features
edges = features.band_edges(y)               # [0.3, 3.99, 6.75, 20.9]
features.price_to_band([2.0, 5.0, 25.0], edges)   # ['Low','Medium','High']
```

## Models & the pipeline

```python
from car_pricing import models
from car_pricing.pipeline import make_pipeline

list(models.model_zoo())                     # the candidates in the bake-off
pipe = make_pipeline(models.model_zoo()["LightGBM"])   # preprocessor + model in one object
```

## Train end-to-end

```python
from car_pricing.train import main as train
result = train()                             # CV bake-off -> pick servable winner -> KPI gate -> save
result["winner"], result["metrics"]["price_mae_lakhs"]   # e.g. ('LightGBM', 0.66)
```

## Config — the single source of truth

```python
from car_pricing import config
config.FEATURES        # the 16 feature names, in order
config.KPI             # {'max_mae_lakhs': 1.0, 'min_r2': 0.85, 'min_band_accuracy': 0.70}
config.BAND_LABELS     # ['Low', 'Medium', 'High']
```

> Want to *extend* these APIs (validation, tuning, drift…)? The stubs in
> [`tests/test_stubs.py`](../tests/test_stubs.py) guide you, and the sibling
> **[app_car_prices_mlops](https://github.com/kalyancn4u/app_car_prices_mlops)** has the
> fully-worked reference implementations.
