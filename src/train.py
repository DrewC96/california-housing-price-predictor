from pathlib import Path

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from src.data import load_housing_data, split_data
from src.features import (
    build_preprocessing,
    drop_capped_labels,
    split_features_labels,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

REGRESSORS = {
    "linear_regression": lambda: LinearRegression(),
    "decision_tree": lambda: DecisionTreeRegressor(random_state=42),
    "random_forest": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
}


def build_model(name: str):
    """Preprocessing and the chosen regressor chained into one object."""
    return make_pipeline(build_preprocessing(), REGRESSORS[name]())


def train():
    housing = load_housing_data()
    train_set, _ = split_data(housing)  # test set stays untouched until evaluate.py
    train_set = drop_capped_labels(train_set)
    X_train, y_train = split_features_labels(train_set)

    results = {}
    for name in REGRESSORS:
        model = build_model(name)
        model.fit(X_train, y_train)
        train_rmse = root_mean_squared_error(y_train, model.predict(X_train))

        cv_rmses = -cross_val_score(
            build_model(name),
            X_train,
            y_train,
            scoring="neg_root_mean_squared_error",
            cv=10,
            n_jobs=-1,
        )
        results[name] = (model, train_rmse, cv_rmses.mean())
        print(
            f"{name:<18} train RMSE: {train_rmse:>9,.0f}   "
            f"CV RMSE: {cv_rmses.mean():>9,.0f} (std {cv_rmses.std():,.0f})"
        )

    best_name = min(results, key=lambda n: results[n][2])
    print(f"\nBest by CV RMSE: {best_name}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(results[best_name][0], BEST_MODEL_PATH)
    print(f"Saved to {BEST_MODEL_PATH}")


if __name__ == "__main__":
    train()