import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "median_house_value"
CAPPED_VALUE = 500001  # the dataset clips house values at this number

# Columns that get a plain impute + scale
PLAIN_NUMERIC = ["longitude", "latitude", "housing_median_age"]
# Heavy-tailed columns that get a log transform first
LOG_FEATURES = [
    "total_bedrooms",
    "total_rooms",
    "population",
    "households",
    "median_income",
]
CATEGORICAL_FEATURES = ["ocean_proximity"]


def drop_capped_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Remove districts whose price was clipped at the dataset's maximum."""
    return df[df[TARGET] < CAPPED_VALUE]


def split_features_labels(df: pd.DataFrame):
    """Separate the inputs (X) from the target (y)."""
    X = df.drop(columns=TARGET)
    y = df[TARGET].copy()
    return X, y


def column_ratio(X):
    """First column divided by the second."""
    X = np.asarray(X)
    return X[:, [0]] / X[:, [1]]


def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]


def make_ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler(),
    )


def make_log_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler(),
    )


def build_preprocessing() -> ColumnTransformer:
    """Impute, engineer ratio features, log-transform, scale, and encode.

    Fit on training data only.
    """
    plain_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )
    preprocessing = ColumnTransformer(
        [
            ("bedrooms_ratio", make_ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            ("rooms_per_house", make_ratio_pipeline(), ["total_rooms", "households"]),
            ("people_per_house", make_ratio_pipeline(), ["population", "households"]),
            ("log", make_log_pipeline(), LOG_FEATURES),
            ("plain", plain_pipeline, PLAIN_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ]
    )
    preprocessing.set_output(transform="pandas")
    return preprocessing


if __name__ == "__main__":
    from src.data import load_housing_data, split_data

    housing = load_housing_data()
    train_set, test_set = split_data(housing)

    print("Train rows before dropping capped labels:", len(train_set))
    train_set = drop_capped_labels(train_set)
    print("Train rows after:", len(train_set))

    X_train, y_train = split_features_labels(train_set)
    preprocessing = build_preprocessing()
    X_train_prepared = preprocessing.fit_transform(X_train)

    print(X_train_prepared.shape)
    print(X_train_prepared.isna().sum().sum(), "missing values left")
    print(X_train_prepared.columns.tolist())