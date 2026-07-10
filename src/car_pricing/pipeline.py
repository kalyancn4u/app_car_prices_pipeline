"""Assemble a preprocessing + model sklearn Pipeline.

Bundling preprocessing and the estimator into ONE object is the central
production-grade decision: the exact same target-encoding that was fitted on the
training data is re-applied at inference automatically, so there is no
possibility of train/serve skew. Ship one `.pkl`, call `.predict()`.
"""

from __future__ import annotations

from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline

from .features import build_preprocessor


def make_pipeline(estimator: RegressorMixin) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ])
