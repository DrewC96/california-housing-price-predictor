from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import root_mean_squared_error

from src.data import load_housing_data, split_data
from src.features import CAPPED_VALUE, TARGET, split_features_labels

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "tuned_random_forest.joblib"
FIGURES_DIR = ROOT / "reports" / "figures"


def rmse_with_ci(y_true, y_pred, confidence: float = 0.95):
    """RMSE plus a confidence interval, computed from the squared errors."""
    squared_errors = (np.asarray(y_pred) - np.asarray(y_true)) ** 2
    low, high = stats.t.interval(
        confidence,
        len(squared_errors) - 1,
        loc=squared_errors.mean(),
        scale=stats.sem(squared_errors),
    )
    return np.sqrt(squared_errors.mean()), np.sqrt(low), np.sqrt(high)


def plot_feature_importances(model, top_n: int = 15):
    preprocessing, forest = model[0], model[-1]
    importances = pd.Series(
        forest.feature_importances_,
        index=preprocessing.get_feature_names_out(),
    ).sort_values(ascending=False)

    print("\nTop feature importances:")
    print(importances.head(top_n).round(3).to_string())

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.head(top_n).iloc[::-1].plot.barh(ax=ax)
    ax.set_xlabel("Importance")
    ax.set_title("Random forest feature importances")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importances.png", dpi=150)
    plt.close(fig)


def plot_predicted_vs_actual(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=4, alpha=0.3)
    limits = [0, max(y_true.max(), y_pred.max())]
    ax.plot(limits, limits, color="red", linewidth=1)  # perfect predictions
    ax.set_xlabel("Actual median house value ($)")
    ax.set_ylabel("Predicted median house value ($)")
    ax.set_title("Predicted vs. actual (test set)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)


def evaluate():
    model = joblib.load(MODEL_PATH)

    housing = load_housing_data()
    _, test_set = split_data(housing)  # identical split: random_state is fixed
    X_test, y_test = split_features_labels(test_set)
    predictions = model.predict(X_test)

    rmse, low, high = rmse_with_ci(y_test, predictions)
    print(f"Test RMSE (all {len(y_test)} districts): ${rmse:,.0f}")
    print(f"95% confidence interval: ${low:,.0f} to ${high:,.0f}")

    # The model was trained without capped prices, so also report the
    # test set with those districts removed.
    uncapped = (y_test < CAPPED_VALUE).to_numpy()
    rmse_uncapped = root_mean_squared_error(y_test[uncapped], predictions[uncapped])
    print(f"Test RMSE (excluding {(~uncapped).sum()} capped districts): ${rmse_uncapped:,.0f}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_feature_importances(model)
    plot_predicted_vs_actual(y_test, predictions)
    print(f"\nFigures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    evaluate()