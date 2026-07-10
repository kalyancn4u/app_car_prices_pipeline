"""End-to-end training: compare models, select the best, evaluate, and ship.

Run:  python -m car_pricing.train      (from src/)  or  make train

Steps
-----
1. Load + clean the raw data.
2. Split into train/test (held-out test never touches model selection).
3. Cross-validate every candidate in the model zoo on the TRAIN set (MAE).
4. Select the winner by CV MAE, refit it on the full train set.
5. Evaluate the winner on the held-out TEST set (R2 / MAE / RMSE / band acc).
6. Gate against the business KPIs, then persist the pipeline + JSON reports.
"""

from __future__ import annotations

import json
import time
from typing import Dict

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from . import config, data, features, models
from .pipeline import make_pipeline

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


def _line(char: str = "-") -> None:
    print(char * 72)


def cross_validate_zoo(X_tr, y_tr) -> Dict[str, Dict[str, float]]:
    """K-fold CV MAE (Lakhs) for every candidate, timed.

    Preprocessing is fit *inside* each fold (fit the target-encoder on the fold's
    train rows, transform its validation rows) so there is no target leakage, and
    the raw estimators are fit/predicted directly on the encoded arrays. Doing it
    manually — rather than via cross_val_score over a Pipeline — keeps every model
    family comparable and side-steps a known xgboost 2.1 x scikit-learn 1.6
    Pipeline tag incompatibility that would otherwise score XGBoost as NaN.
    """
    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    zoo = models.model_zoo()
    results: Dict[str, Dict[str, float]] = {}
    print(f"\nCross-validating {len(zoo)} models "
          f"({config.CV_FOLDS}-fold, MAE in Lakhs - lower is better):\n")
    print(f"  {'Model':<24}{'CV MAE':>10}{'+/-std':>9}{'time(s)':>10}")
    _line()
    for name, base in zoo.items():
        maes = []
        t = time.perf_counter()
        for tr_idx, val_idx in cv.split(X_tr):
            Xt, Xv = X_tr.iloc[tr_idx], X_tr.iloc[val_idx]
            yt, yv = y_tr.iloc[tr_idx], y_tr.iloc[val_idx]
            pre = features.build_preprocessor()
            Xt_e = pre.fit_transform(Xt, yt)
            Xv_e = pre.transform(Xv)
            est = clone(base)
            est.fit(Xt_e, yt)
            maes.append(mean_absolute_error(yv, est.predict(Xv_e)))
        arr = np.array(maes)
        elapsed = time.perf_counter() - t
        results[name] = {"cv_mae": float(arr.mean()),
                         "cv_std": float(arr.std()),
                         "cv_time_s": elapsed}
        print(f"  {name:<24}{arr.mean():>10.3f}{arr.std():>9.3f}{elapsed:>10.1f}")
    return results


def main() -> Dict:
    _line("=")
    print("Car Price MLOps - training pipeline")
    _line("=")

    df = data.clean(data.load_raw())
    print(f"Loaded + cleaned: {len(df):,} rows "
          f"(dropped {df.attrs.get('dropped_rows', 0):,}).")

    X, y = features.split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    edges = features.band_edges(y_tr)
    print(f"Train/test: {len(X_tr):,} / {len(X_te):,}.  Band edges (Lakhs): {edges}")

    comparison = cross_validate_zoo(X_tr, y_tr)
    ranked = sorted(comparison, key=lambda k: comparison[k]["cv_mae"])
    cv_best = ranked[0]
    print(f"\n>>> Best CV MAE: {cv_best} ({comparison[cv_best]['cv_mae']:.3f} Lakhs)")

    # Ship the best model that also SERVES cleanly in a portable sklearn
    # Pipeline. The installed xgboost 2.1 x scikit-learn 1.6 combo can score
    # XGBoost during CV but cannot serialise/serve it through a Pipeline, so if
    # it ranks first we transparently fall through to the next-best model that
    # deploys without a library-version shim — accuracy is a near-tie anyway.
    winner = best = pred = fit_s = None
    for name in ranked:
        candidate = make_pipeline(models.model_zoo()[name])
        try:
            t = time.perf_counter()
            candidate.fit(X_tr, y_tr)
            fs = time.perf_counter() - t
            p = candidate.predict(X_te)
        except Exception as exc:  # not servable in a Pipeline here
            print(f"    [skip] {name}: best-ranked but not servable in a Pipeline "
                  f"({type(exc).__name__}); trying next-best.")
            continue
        winner, best, pred, fit_s = name, candidate, p, fs
        break
    if best is None:
        raise RuntimeError("No candidate could be fit and served in a Pipeline.")
    note = "" if winner == cv_best else f"  (CV-best '{cv_best}' isn't cleanly servable here)"
    print(f">>> Shipped model: {winner}{note}")

    true_band = features.price_to_band(y_te, edges)
    pred_band = features.price_to_band(pred, edges)
    metrics = {
        "winner": winner,
        "cv_best_model": cv_best,
        "cv_best_mae_lakhs": round(comparison[cv_best]["cv_mae"], 3),
        "price_r2": round(float(r2_score(y_te, pred)), 4),
        "price_mae_lakhs": round(float(mean_absolute_error(y_te, pred)), 3),
        "price_rmse_lakhs": round(float(np.sqrt(mean_squared_error(y_te, pred))), 3),
        "band_accuracy": round(float((true_band == pred_band).mean()), 4),
        "fit_seconds": round(fit_s, 2),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_features": len(config.FEATURES),
    }

    _line("=")
    print("Held-out test performance (winner):")
    print(f"  R^2            : {metrics['price_r2']:.4f}")
    print(f"  MAE            : Rs {metrics['price_mae_lakhs']:.3f} Lakhs")
    print(f"  RMSE           : Rs {metrics['price_rmse_lakhs']:.3f} Lakhs")
    print(f"  Band accuracy  : {metrics['band_accuracy'] * 100:.2f}%")

    # KPI gate
    gate = {
        "mae_ok": metrics["price_mae_lakhs"] <= config.KPI["max_mae_lakhs"],
        "r2_ok": metrics["price_r2"] >= config.KPI["min_r2"],
        "band_ok": metrics["band_accuracy"] >= config.KPI["min_band_accuracy"],
    }
    gate["all_pass"] = all(gate.values())
    metrics["kpi_gate"] = gate
    print(f"  KPI gate       : {'PASS' if gate['all_pass'] else 'FAIL'}  {gate}")
    _line("=")

    # Persist artifacts.
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if joblib is not None:
        joblib.dump(best, config.PIPELINE_PATH, compress=("lzma", 9))
        metrics["artifact_mb"] = round(config.PIPELINE_PATH.stat().st_size / 1024 / 1024, 2)
        print(f"Saved pipeline -> {config.PIPELINE_PATH.name} "
              f"({metrics['artifact_mb']} MB)")

    meta = features.build_serving_metadata(df, edges)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    config.COMPARISON_PATH.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    config.METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved metrics, comparison and serving metadata to {config.MODELS_DIR.name}/.")

    return {"metrics": metrics, "comparison": comparison, "winner": winner}


if __name__ == "__main__":
    main()
