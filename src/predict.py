import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.data import load_housing_data, split_data
from src.features import (
    CATEGORICAL_FEATURES,
    LOG_FEATURES,
    PLAIN_NUMERIC,
    TARGET,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "tuned_random_forest.joblib"
REQUIRED_COLUMNS = PLAIN_NUMERIC + LOG_FEATURES + CATEGORICAL_FEATURES


def load_model(path: Path = MODEL_PATH):
    if not path.is_file():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python -m src.tune` first."
        )
    return joblib.load(path)


def predict_prices(model, df: pd.DataFrame) -> pd.Series:
    """Predict median house values for raw, unprocessed district rows."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    X = df.drop(columns=[TARGET], errors="ignore")
    return pd.Series(model.predict(X), index=df.index, name="predicted_value")


def main():
    parser = argparse.ArgumentParser(description="Predict California house prices.")
    parser.add_argument(
        "csv",
        nargs="?",
        help="CSV of districts with the same feature columns as housing.csv. "
        "If omitted, predicts 5 districts from the test set as a demo.",
    )
    parser.add_argument("--output", help="Optional path to save predictions as CSV.")
    args = parser.parse_args()

    model = load_model()

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        _, test_set = split_data(load_housing_data())
        df = test_set.head(5)

    result = df.copy()
    result["predicted_value"] = predict_prices(model, df).round(0)

    if TARGET in result.columns:
        print(result[[TARGET, "predicted_value"]].to_string())
    else:
        print(result["predicted_value"].to_string())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nSaved predictions to {args.output}")


if __name__ == "__main__":
    main()