import numpy as np
import pandas as pd
import pytest

from src.features.engineer import (
    add_card_aggregates,
    add_email_features,
    add_time_features,
)


def make_sample_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "TransactionDT": rng.integers(0, 86_400 * 30, size=n),
        "TransactionAmt": rng.uniform(10, 500, size=n),
        "card1": rng.integers(1000, 9999, size=n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], size=n),
        "R_emaildomain": rng.choice(["gmail.com", "hotmail.com", None], size=n),
    })


def test_time_features_range():
    df = add_time_features(make_sample_df())
    assert "tx_hour" in df.columns
    assert "tx_day" in df.columns
    assert df["tx_hour"].between(0, 23).all()
    assert df["tx_day"].between(0, 6).all()


def test_card_aggregates_no_nulls():
    df = add_time_features(make_sample_df())
    df = add_card_aggregates(df)
    assert df["card_daily_tx_count"].isna().sum() == 0
    assert df["tx_to_card_mean_ratio"].isna().sum() == 0
    assert "_card_day" not in df.columns  # internal column must be dropped


def test_email_feature_is_binary():
    df = add_email_features(make_sample_df())
    assert set(df["same_email_domain"].unique()).issubset({0, 1})


def test_no_mutation_of_input():
    df = make_sample_df()
    original_cols = list(df.columns)
    add_time_features(df)
    assert list(df.columns) == original_cols
