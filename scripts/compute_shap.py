"""
Compute SHAP values for the trained XGBoost model and save a summary plot.

Usage:
    python scripts/compute_shap.py

Outputs:
    figures/shap_summary.png
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import shap

from src.data.loader import load_raw
from src.features.engineer import build_features
from src.models.train import get_feature_cols, time_based_split


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


def main():
    print("Loading data...")
    df = load_raw()
    df = build_features(df)

    _, val_df = time_based_split(df)
    feature_cols = get_feature_cols(df)

    preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.pkl")
    model = joblib.load(ARTIFACTS_DIR / "xgboost.pkl")

    X_val = preprocessor.transform(val_df[feature_cols])

    sample = X_val[:5_000]

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    FIGURES_DIR.mkdir(exist_ok=True)
    shap.summary_plot(shap_values, sample, feature_names=feature_cols, show=False)
    out_path = FIGURES_DIR / "shap_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
