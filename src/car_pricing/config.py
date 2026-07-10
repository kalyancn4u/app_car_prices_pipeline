"""Central configuration — the single source of truth for the whole pipeline.

Every phase (notebooks, training, serving) imports from here so there is exactly
one definition of the schema, the split, the KPI targets and the artifact paths.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
# Stored gzip-compressed (~81% smaller); pandas.read_csv infers .gz and
# decompresses on read, so nothing else in the pipeline needs to change.
DATA_RAW = ROOT / "data" / "raw" / "cars24-car-price-cleaned-new.csv.gz"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

PIPELINE_PATH = MODELS_DIR / "price_pipeline.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
COMPARISON_PATH = MODELS_DIR / "model_comparison.json"
METADATA_PATH = MODELS_DIR / "serving_metadata.json"

# --- Schema ----------------------------------------------------------------
TARGET = "selling_price"          # in Lakhs of Rupees (e.g. 4.75 == Rs 4.75 L)
PRICE_UNIT = "Lakhs"

# High-cardinality categoricals -> TARGET-ENCODED (not one-hot). This is the
# core production insight: `model` has 3,233 levels; one-hot would explode the
# feature matrix to ~3,200 columns and bloat every model. Target encoding maps
# each level to its (cross-fitted) mean price -> two compact numeric columns.
TARGET_ENCODE = ["make", "model"]

NUMERIC = ["km_driven", "mileage", "engine", "max_power", "age"]

# Already one-hot in the source data, with a dropped baseline per group
# (Dealer / CNG / Automatic / <5 seats). Passed through as 0/1 flags.
FLAGS = [
    "Individual", "Trustmark Dealer",
    "Diesel", "Electric", "LPG", "Petrol",
    "Manual", "Seats_5", "Seats_Above_5",
]

FEATURES = TARGET_ENCODE + NUMERIC + FLAGS

# --- Price bands (terciles -> Low / Medium / High) -------------------------
BAND_QUANTILES = [0.0, 1 / 3, 2 / 3, 1.0]
BAND_LABELS = ["Low", "Medium", "High"]

# --- Train / test / CV -----------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 3

# --- Business KPI targets (see docs/BUSINESS_CASE.md) ----------------------
# The model is "good enough to ship" only if it clears these thresholds.
KPI = {
    "max_mae_lakhs": 1.0,        # typical error must be under Rs 1,00,000
    "min_r2": 0.85,              # explain at least 85% of price variance
    "min_band_accuracy": 0.70,   # correct budget band at least 70% of the time
}
