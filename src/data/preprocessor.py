import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


CATEGORICAL_COLS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Fill numeric NaN with training median; add binary missing-indicator columns."""

    def fit(self, X: pd.DataFrame, y=None):
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.medians_ = X[self.numeric_cols_].median()
        self.cols_with_na_ = [c for c in self.numeric_cols_ if X[c].isna().any()]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols_with_na_:
            X[f"{col}_missing"] = X[col].isna().astype(np.int8)
        X[self.numeric_cols_] = X[self.numeric_cols_].fillna(self.medians_)
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Label-encode categorical columns; unseen labels at inference map to 'unknown'."""

    def __init__(self, categorical_cols: list = None):
        self.categorical_cols = categorical_cols or CATEGORICAL_COLS

    def fit(self, X: pd.DataFrame, y=None):
        self.encoders_: dict = {}
        for col in self.categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                le.fit(X[col].fillna("unknown").astype(str))
                self.encoders_[col] = le
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, le in self.encoders_.items():
            if col in X.columns:
                known = set(le.classes_)
                values = X[col].fillna("unknown").astype(str)
                values = values.apply(lambda v: v if v in known else "unknown")
                X[col] = le.transform(values)
        return X


def build_preprocessor() -> Pipeline:
    return Pipeline([
        ("missing", MissingValueHandler()),
        ("categorical", CategoricalEncoder()),
    ])
