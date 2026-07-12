"""Feature engineering: the preprocessor, the price bands, and serving metadata.

The preprocessor is the heart of the production design. Rather than one-hot
encoding `make`/`model` (3,233 levels -> a ~3,200-column, 100 MB+ model), we
TARGET-ENCODE them into two compact columns. sklearn's TargetEncoder does this
with internal cross-fitting, so there is no target leakage into the training
folds. The result: a ~16-feature dense matrix every model family can consume
(including HistGradientBoosting, which cannot take sparse input).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder

from . import config


def build_preprocessor() -> ColumnTransformer:
    """Target-encode make/model; pass numeric + flag columns through unchanged."""
    return ColumnTransformer(
        transformers=[
            ("target_enc",
             TargetEncoder(target_type="continuous", random_state=config.RANDOM_STATE),
             config.TARGET_ENCODE),
            ("passthrough", "passthrough", config.NUMERIC + config.FLAGS),
        ],
        remainder="drop",
    )


def band_edges(y: pd.Series) -> List[float]:
    """Tercile cut points (in Lakhs) that define Low / Medium / High."""
    edges = [float(y.quantile(q)) for q in config.BAND_QUANTILES]
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 0.01
    return [round(e, 2) for e in edges]


def price_to_band(prices, edges: List[float]) -> np.ndarray:
    """Map price(s) in Lakhs to a Low/Medium/High label using the tercile edges.

    Deriving the band from the (very accurate) predicted price guarantees the
    band and the rupee figure never disagree — a deliberate improvement over
    training a separate, less-accurate band classifier.
    """
    prices = np.asarray(prices, dtype=float)
    # np.digitize with the two internal edges -> index 0/1/2 -> label.
    idx = np.clip(np.digitize(prices, edges[1:-1]), 0, len(config.BAND_LABELS) - 1)
    return np.array(config.BAND_LABELS)[idx]


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned DataFrame into the ordered feature matrix X and the target y."""
    return df[config.FEATURES].copy(), df[config.TARGET].copy()


def build_serving_metadata(df: pd.DataFrame, edges: List[float]) -> Dict:
    """Everything an API/UI needs to offer valid inputs and auto-fill specs.

    Mirrors the contract used by the companion Streamlit/Flask projects: the
    make->model map, per-model median specs and observed categorical options.
    """
    makes_models = {
        make: sorted(g["model"].unique().tolist())
        for make, g in df.groupby("make")
    }
    spec_cols = ["mileage", "engine", "max_power"]
    model_specs: Dict[str, Dict] = {}
    for (make, model), row in df.groupby(["make", "model"])[spec_cols].median().round(1).iterrows():
        model_specs.setdefault(make, {})[model] = {c: float(row[c]) for c in spec_cols}
    make_specs = {
        make: {c: float(v) for c, v in row.items()}
        for make, row in df.groupby("make")[spec_cols].median().round(1).iterrows()
    }
    return {
        "price_unit": config.PRICE_UNIT,
        "n_samples": int(len(df)),
        "band_labels": config.BAND_LABELS,
        "band_edges": edges,
        "makes_models": dict(sorted(makes_models.items())),
        "model_specs": model_specs,
        "make_specs": make_specs,
        "numeric_defaults": {c: float(df[c].median()) for c in config.NUMERIC},
    }
