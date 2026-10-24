import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_URL = "https://github.com/ageron/data/raw/main/housing.tgz"

def load_housing_data() -> pd.DataFrame:
    """Download the housing dataset if needed and return it as a DataFrame."""
    tarball_path = DATA_DIR / "housing.tgz"
    if not tarball_path.is_file():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, tarball_path)
        with tarfile.open(tarball_path) as housing_tar:
            housing_tar.extractall(path=DATA_DIR, filter="data")
    return pd.read_csv(DATA_DIR / "housing" / "housing.csv")


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified train/test split on median income category."""
    income_cat = pd.cut(
        df["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    strat_train_set, strat_test_set = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=income_cat
    )
    return strat_train_set, strat_test_set


if __name__ == "__main__":
    housing = load_housing_data()
    housing.info()
    train_set, test_set = split_data(housing)
    print(f"Train: {len(train_set)}, Test: {len(test_set)}")