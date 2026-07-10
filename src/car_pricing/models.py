"""The candidate model zoo for the bake-off (notebook 05 / train.py).

Every candidate is a *regressor* predicting the exact price in Lakhs; the
Low/Medium/High band is derived from the predicted price downstream, so no
separate classifier is trained. Hyper-parameters are deliberately modest and
comparable across families — the point is a fair comparison, not squeezing the
last 0.1% out of any single model.
"""

from __future__ import annotations

from typing import Dict

from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from . import config

RS = config.RANDOM_STATE


def model_zoo() -> Dict[str, RegressorMixin]:
    """Name -> unfitted estimator. XGBoost/LightGBM added only if importable."""
    zoo: Dict[str, RegressorMixin] = {
        "Ridge (linear)": Ridge(alpha=1.0, random_state=RS),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=8, random_state=RS),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=18, min_samples_leaf=4, n_jobs=-1, random_state=RS),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, max_depth=8, random_state=RS),
    }

    # NOTE: on xgboost 2.1.x x scikit-learn 1.6.x, XGBoost fits inside a Pipeline
    # but cannot `.predict()` through it (a tag-system incompatibility, fixed in
    # xgboost>=2.1.2). train.py handles this by shipping the best *servable* model;
    # the full analysis + head-to-head is in docs/XGBOOST_SERVABILITY.md.
    try:
        from xgboost import XGBRegressor
        zoo["XGBoost"] = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
            random_state=RS, verbosity=0,
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor
        zoo["LightGBM"] = LGBMRegressor(
            n_estimators=600, num_leaves=63, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
            random_state=RS, verbosity=-1,
        )
    except ImportError:
        pass

    return zoo
