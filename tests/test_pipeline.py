"""Smoke + contract tests for the production pipeline.

Run with:  pytest -q   (needs the trained artifacts under models/ — run
`python -m car_pricing.train` once if they're missing).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from car_pricing import config, data, features

ARTIFACTS = config.PIPELINE_PATH.exists() and config.METRICS_PATH.exists()
needs_model = pytest.mark.skipif(not ARTIFACTS, reason="run `python -m car_pricing.train` first")


# --- Data + feature contracts (no trained model needed) --------------------
def test_clean_produces_expected_schema():
    df = data.clean(data.load_raw())
    for col in config.FEATURES + [config.TARGET]:
        assert col in df.columns, f"missing {col}"
    assert (df[config.TARGET] > 0).all()


def test_band_edges_and_mapping_are_consistent():
    df = data.clean(data.load_raw())
    edges = features.band_edges(df[config.TARGET])
    assert edges == sorted(edges), "edges must be increasing"
    bands = features.price_to_band([edges[0], edges[1] + 0.1, edges[-1]], edges)
    assert list(bands) == ["Low", "Medium", "High"]


def test_format_roundtrip_preserves_data(tmp_path: Path):
    df = data.clean(data.load_raw()).head(500)
    for fname in ("t.csv", "t.csv.gz", "t.parquet", "t.feather"):
        p = tmp_path / fname
        data.write_dataframe(df, p)
        back = data.read_dataframe(p)
        assert back.shape == df.shape


# --- Model / serving contracts (need the trained artifacts) ----------------
@needs_model
def test_metrics_pass_kpi_gate():
    metrics = json.loads(config.METRICS_PATH.read_text())
    assert metrics["kpi_gate"]["all_pass"], metrics["kpi_gate"]


@needs_model
def test_predict_returns_sensible_price_and_band():
    from car_pricing.predict import predict
    out = predict({"make": "MARUTI", "model": "SWIFT VXI", "age": 5, "km_driven": 40000})
    assert 1 < out["predicted_price_lakhs"] < 50
    assert out["price_band"] in {"Low", "Medium", "High"}


@needs_model
def test_premium_car_costs_more_than_hatchback():
    from car_pricing.predict import predict
    swift = predict({"make": "MARUTI", "model": "SWIFT VXI"})["predicted_price_lakhs"]
    bmw = predict({"make": "BMW", "model": "X5"})["predicted_price_lakhs"]
    assert bmw > swift * 2, (swift, bmw)
