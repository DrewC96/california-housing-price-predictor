from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline

from src.data import load_housing_data, split_data
from src.features import (
    build_preprocessing,
    drop_capped_labels,
    split_features_labels,
)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
TUNED_MODEL_PATH = MODELS_DIR / "tuned_random_forest.joblib"
REPORTS_DIR = ROOT / "reports"

# Pipeline step names are lowercase class names: "randomforestregressor"
PARAM_DISTRIBUTIONS = {
    "randomforestregressor__n_estimators": randint(100, 300),
    "randomforestregressor__max_features": randint(2, 16),
    "randomforestregressor__max_depth": [None, 10, 20, 30],
    "randomforestregressor__min_samples_leaf": randint(1, 10),
}


def tune():
    housing = load_housing_data()
    train_set, _ = split_data(housing)  # test set stays untouched
    train_set = drop_capped_labels(train_set)
    X_train, y_train = split_features_labels(train_set)

    pipeline = make_pipeline(
        build_preprocessing(),
        RandomForestRegressor(random_state=42),
    )

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=20,
        cv=5,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"\nBest CV RMSE: {-search.best_score_:,.0f}")
    print("Best parameters:")
    for name, value in search.best_params_.items():
        print(f"  {name.split('__')[1]}: {value}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(search.best_estimator_, TUNED_MODEL_PATH)
    print(f"Saved to {TUNED_MODEL_PATH}")

    REPORTS_DIR.mkdir(exist_ok=True)
    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    results.to_csv(REPORTS_DIR / "tuning_results.csv", index=False)


if __name__ == "__main__":
    tune()