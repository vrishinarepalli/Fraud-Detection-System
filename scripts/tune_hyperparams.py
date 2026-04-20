"""
Optuna hyperparameter search for XGBoost.

Usage:
    python scripts/tune_hyperparams.py --n-trials 50

Outputs:
    artifacts/best_params.yaml  (git-ignored -- do not commit)
"""
import argparse
from pathlib import Path

import optuna
import yaml
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.data.loader import load_raw
from src.data.preprocessor import build_preprocessor
from src.features.engineer import build_features
from src.models.train import get_feature_cols, time_based_split


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "auc",
        "n_jobs": -1,
        "random_state": 42,
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])


def main(n_trials: int):
    print("Loading data...")
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

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight),
        n_trials=n_trials,
    )

    print(f"\nBest AUC-ROC: {study.best_value:.4f}")
    print("Best params:", study.best_params)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    out_path = ARTIFACTS_DIR / "best_params.yaml"
    with open(out_path, "w") as f:
        yaml.dump(study.best_params, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()
    main(args.n_trials)
