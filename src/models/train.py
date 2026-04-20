from pathlib import Path

import joblib
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.data.loader import load_raw
from src.data.preprocessor import build_preprocessor
from src.features.engineer import build_features


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "model_config.yaml"
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def time_based_split(df: pd.DataFrame, val_frac: float = 0.2):
    """Split by TransactionDT so validation is always later in time than training."""
    df_sorted = df.sort_values("TransactionDT")
    split_idx = int(len(df_sorted) * (1 - val_frac))
    return df_sorted.iloc[:split_idx].copy(), df_sorted.iloc[split_idx:].copy()


def get_feature_cols(df: pd.DataFrame) -> list:
    drop = {"TransactionID", "TransactionDT", "isFraud"}
    return [c for c in df.columns if c not in drop]


def train(config_path: Path = CONFIG_PATH):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("Loading and engineering features...")
    df = load_raw()
    df = build_features(df)

    train_df, val_df = time_based_split(df)
    feature_cols = get_feature_cols(df)

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df[feature_cols])
    y_train = train_df["isFraud"].values
    X_val = preprocessor.transform(val_df[feature_cols])
    y_val = val_df["isFraud"].values

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", **cfg.get("logistic_regression", {})
        ),
        "xgboost": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            **cfg.get("xgboost", {}),
        ),
        "lightgbm": LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            **cfg.get("lightgbm", {}),
        ),
    }

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        results[name] = auc
        print(f"  {name}: AUC-ROC = {auc:.4f}")
        joblib.dump(model, ARTIFACTS_DIR / f"{name}.pkl")

    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.pkl")

    best = max(results, key=results.get)
    print(f"\nBest model: {best} ({results[best]:.4f})")
    return results


if __name__ == "__main__":
    train()
