# 🧩 XGBoost Servability — closing the bake-off thread

In the model comparison ([`MODEL_CARD.md`](MODEL_CARD.md) /
[notebook 05](../notebooks/05_model_comparison.ipynb)) **XGBoost had the best
cross-validated MAE (0.708 L)**, yet the pipeline shipped **LightGBM**. This note
answers the two questions that leaves open — *why* couldn't XGBoost ship, and
*would it even have been better?* — with the full analysis reproducible in
[notebook 08](../notebooks/08_xgboost_deep_dive.ipynb).

## 1 · The symptom

XGBoost fits inside a scikit-learn `Pipeline` fine; the break is at **`.predict()`**:

```
AttributeError: 'super' object has no attribute '__sklearn_tags__'
```

`train.py` catches exactly this and transparently falls through to the next-best
**servable** model — which is why LightGBM ships.

## 2 · Root cause — a broken cooperative-inheritance chain

scikit-learn 1.6 rebuilt estimator introspection around `__sklearn_tags__`, which
each class contributes to **cooperatively** via `super().__sklearn_tags__()`. That
chain only works if the estimator lists the **mixin first and `BaseEstimator`
last**. XGBoost 2.1.1 declares its bases the other way round:

```
XGBRegressor MRO:
  XGBRegressor -> XGBModel -> BaseEstimator -> ... -> RegressorMixin -> object
                              ^^^^^^^^^^^^^              ^^^^^^^^^^^^^^
                              before the mixin          reached too late
```

Because `BaseEstimator` precedes `RegressorMixin`, `RegressorMixin`'s
`super().__sklearn_tags__()` resolves to `object`, which has no such method →
`AttributeError`. It is a **library-compatibility bug, not a data or modelling
issue**, and it is fixed in **xgboost ≥ 2.1.2**.

Environment here: `xgboost 2.1.1` × `scikit-learn 1.6.1`.

## 3 · Fix options

| Option | What you do | Verdict |
| :----- | :---------- | :------ |
| **Upgrade (recommended)** | `pip install 'xgboost>=2.1.2'` (or pin `scikit-learn<1.6`) | ✅ Cleanest — XGBoost then drops straight into the same `Pipeline`, and `train.py` would select it automatically. |
| **Decoupled serving** | Persist the fitted `ColumnTransformer` and `XGBRegressor` separately; predict with `xgb.predict(pre.transform(X))` | ✅ Works on any versions; used below to get XGBoost's real test numbers. |
| Monkey-patch `__sklearn_tags__` | Inject the missing method at runtime | ⚠️ Fragile and version-specific — **not** recommended. |

## 4 · Head-to-head (held-out test set)

XGBoost was evaluated via **decoupled serving**; LightGBM figures are the shipped
model's. Same split, same features, same band edges.

| Metric | XGBoost | LightGBM (shipped) |
| :----- | ------: | -----------------: |
| CV MAE (Lakhs) | **0.708** | 0.715 |
| Test MAE (Lakhs) | 0.668 | **0.664** |
| Test R² | 0.9568 | 0.9568 |
| Test RMSE (Lakhs) | 1.021 | 1.022 |
| Band accuracy | 0.8605 | 0.8582 |
| Servable in a portable `Pipeline` (this env) | ❌ no | ✅ yes |

## 5 · Conclusion

- **The two models are statistically indistinguishable on the held-out test set** —
  identical R², MAE within ~₹400, band accuracy within 0.2 pt.
- Tellingly, **LightGBM is marginally *better* on test MAE** despite XGBoost's
  marginally better *CV* MAE — the ranking didn't survive out-of-sample, which
  confirms the gap is noise, not signal.
- **Shipping LightGBM therefore cost zero accuracy** while buying clean, shim-free
  serialisation and serving. This validates the automated "ship the best *servable*
  model" rule in `train.py`.
- **To ship XGBoost instead:** upgrade to `xgboost>=2.1.2` and re-run
  `python -m car_pricing.train` — it will slot into the same `Pipeline` and the
  selector will pick it if it still leads. No code changes required.

The bake-off thread is now fully closed: XGBoost was investigated, evaluated, and
consciously not shipped — for a documented, reproducible reason.
