"""Load the shipped pipeline and turn a car description into a price + band.

This is the serving surface: it mirrors the "friendly" auto-fill contract of the
companion Streamlit/Flask projects — callers give what they know (make, model,
and optionally age/km/...), and anything omitted is filled from that model's
typical values, so no knowledge of the internal feature schema is required.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict

import joblib
import pandas as pd

from . import config, features

_FLAG_MAP = {
    "seller": {"Dealer": [], "Individual": ["Individual"],
               "Trustmark Dealer": ["Trustmark Dealer"]},
    "fuel": {"CNG": [], "Petrol": ["Petrol"], "Diesel": ["Diesel"],
             "Electric": ["Electric"], "LPG": ["LPG"]},
    "transmission": {"Automatic": [], "Manual": ["Manual"]},
    "seats": {"Fewer than 5": [], "5": ["Seats_5"], "More than 5": ["Seats_Above_5"]},
}


@lru_cache(maxsize=1)
def _load():
    pipeline = joblib.load(config.PIPELINE_PATH)
    meta = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
    return pipeline, meta


def format_price(lakhs: float) -> str:
    if lakhs >= 100:
        return f"₹{lakhs / 100:.2f} Crore"
    if lakhs >= 1:
        return f"₹{lakhs:.2f} Lakhs"
    return f"₹{lakhs * 100_000:,.0f}"


def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Predict price (Lakhs) + budget band for a car described by `payload`."""
    pipeline, meta = _load()

    make = str(payload["make"]).strip().upper()
    model = str(payload["model"]).strip().upper()
    specs = (meta["model_specs"].get(make, {}).get(model)
             or meta["make_specs"].get(make)
             or {k: meta["numeric_defaults"][k] for k in ("mileage", "engine", "max_power")})

    row = {c: 0 for c in config.FEATURES}
    row["make"], row["model"] = make, model
    row["age"] = float(payload.get("age", meta["numeric_defaults"]["age"]))
    row["km_driven"] = float(payload.get("km_driven", meta["numeric_defaults"]["km_driven"]))
    row["mileage"] = float(payload.get("mileage", specs["mileage"]))
    row["engine"] = float(payload.get("engine", specs["engine"]))
    row["max_power"] = float(payload.get("max_power", specs["max_power"]))
    for group, default in (("seller", "Dealer"), ("fuel", "Petrol"),
                           ("transmission", "Manual"), ("seats", "5")):
        choice = str(payload.get(group, default))
        for flag in _FLAG_MAP[group].get(choice, []):
            row[flag] = 1

    X = pd.DataFrame([row])[config.FEATURES]
    price = max(float(pipeline.predict(X)[0]), 0.0)
    band = str(features.price_to_band([price], meta["band_edges"])[0])
    return {
        "predicted_price_lakhs": round(price, 2),
        "predicted_price_display": format_price(price),
        "price_band": band,
    }
