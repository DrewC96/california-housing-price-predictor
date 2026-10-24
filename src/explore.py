from pathlib import Path

import matplotlib.pyplot as plt

from src.data import load_housing_data

FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"


def plot_histograms(housing):
    housing.hist(bins=50, figsize=(12, 8))
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "histograms.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    housing = load_housing_data()
    plot_histograms(housing)