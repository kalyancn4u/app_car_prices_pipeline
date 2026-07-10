"""Guided test stubs — your turn to complete them. 🧑‍🎓

Every test below is intentionally **skipped**. To complete one:

    1. Read its docstring (it says exactly what to check).
    2. Delete the `@pytest.mark.skip(...)` line above it.
    3. Replace the `pytest.fail(...)` body with real `assert` statements.
    4. Run `pytest -v` until it turns green.

Run them any time with:

    pytest -v            # the stubs show as SKIPPED; tests/test_pipeline.py passes

The working reference tests in ``tests/test_pipeline.py`` show the patterns and the
imports you'll need. The stubs are ordered easy → advanced; the last group targets the
"pending" MLOps stages, so completing them means genuinely extending the project.
"""

import pytest

# Everything the stubs are likely to need (already used by test_pipeline.py):
from car_pricing import config, data, features   # noqa: F401  (used once you implement)

TODO = "stub — delete this @skip, then implement the body (see the docstring)"


# ===========================================================================
# GROUP 1 · Data  (pandas basics)
# ===========================================================================

@pytest.mark.skip(reason=TODO)
def test_clean_drops_rows_with_nonpositive_price():
    """`data.clean` must drop rows whose selling_price is 0, negative, or NaN.

    Hint: build a tiny DataFrame with a good row and a few bad ones, pass it through
    `data.clean(df)`, and assert every remaining `selling_price` is > 0.
    """
    pytest.fail("TODO: implement this test")


@pytest.mark.skip(reason=TODO)
def test_clean_uppercases_make_and_model():
    """`data.clean` should normalise `make`/`model` to UPPERCASE and strip spaces.

    Hint: feed in `" maruti "` / `"swift vxi"` and assert they come out as
    `"MARUTI"` / `"SWIFT VXI"`.
    """
    pytest.fail("TODO: implement this test")


# ===========================================================================
# GROUP 2 · Features  (preprocessing)
# ===========================================================================

@pytest.mark.skip(reason=TODO)
def test_feature_matrix_has_expected_column_count():
    """The production feature set should be compact (target-encoding, not one-hot).

    Hint: assert `len(config.FEATURES)` equals the documented count (see MODEL_CARD /
    config.py — it should be 16, not thousands).
    """
    pytest.fail("TODO: implement this test")


@pytest.mark.skip(reason=TODO)
def test_price_to_band_maps_edges_correctly():
    """`features.price_to_band` must map a price onto Low / Medium / High.

    Hint: compute `edges = features.band_edges(y)` on the real target, then assert a
    very low price → "Low", a mid price → "Medium", a very high price → "High".
    """
    pytest.fail("TODO: implement this test")


# ===========================================================================
# GROUP 3 · Serving  (error handling)  — this one is a real EXTENSION
# ===========================================================================

@pytest.mark.skip(reason=TODO)
def test_predict_rejects_unknown_make():
    """An unknown make/model should be reported, not silently guessed.

    Today `car_pricing.predict.predict` falls back to global averages for an unknown
    car. IMPROVE it: validate `make`/`model` against `serving_metadata.json`'s
    `makes_models` and raise `ValueError` for something like
    `{"make": "NOTABRAND", "model": "X"}` — then assert that here with `pytest.raises`.
    """
    pytest.fail("TODO: implement the validation, then this test")


@pytest.mark.skip(reason=TODO)
def test_predict_is_deterministic():
    """The same input must always yield the same prediction.

    Hint: call `predict(payload)` twice with identical input and assert the two
    `predicted_price_lakhs` values are equal.
    """
    pytest.fail("TODO: implement this test")


# ===========================================================================
# GROUP 4 · Modelling  (hyper-parameter search — an extension)
# ===========================================================================

@pytest.mark.skip(reason=TODO)
def test_every_model_in_the_zoo_can_fit_and_predict():
    """Each candidate in `car_pricing.models.model_zoo()` should fit and predict.

    Hint: build the preprocessor, transform a small sample, then loop the zoo,
    `clone` each estimator, `.fit(...)` and `.predict(...)`, asserting the output
    length matches. (See how `train.cross_validate_zoo` does it.)
    """
    pytest.fail("TODO: implement this test")


@pytest.mark.skip(reason=TODO)
def test_tuned_model_matches_or_beats_the_default():
    """A tuned model should do at least as well as the shipped default.

    EXTENSION: add a small hyper-parameter search (e.g. RandomizedSearchCV or Optuna)
    for one model, then assert its cross-validated MAE is <= the default's MAE from
    `models/model_comparison.json` (allow a small tolerance).
    """
    pytest.fail("TODO: add tuning, then this test")


# ===========================================================================
# GROUP 5 · Monitoring & Drift  (the "pending" MLOps stages)
# ===========================================================================

@pytest.mark.skip(reason=TODO)
def test_drift_is_flagged_on_a_shifted_distribution():
    """A drift check should fire when new data looks different from training data.

    EXTENSION: write a small `detect_drift(reference_df, new_df)` (e.g. a KS-test or PSI
    per numeric column, or use `evidently`). Assert it flags NO drift when `new_df`
    is a sample of the training data, and flags drift when you shift a column
    (e.g. add 10 years to every `age`).
    """
    pytest.fail("TODO: build drift detection, then this test")


@pytest.mark.skip(reason=TODO)
def test_predictions_are_logged_for_monitoring():
    """Served predictions should be logged so live accuracy can be tracked later.

    EXTENSION: add optional logging to `predict()` (append input + output to a JSONL
    file), then assert a line is written and round-trips back to the same values.
    """
    pytest.fail("TODO: add prediction logging, then this test")
