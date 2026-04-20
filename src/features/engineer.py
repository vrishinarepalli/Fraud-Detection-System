import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive hour-of-day and day-of-week from TransactionDT (seconds offset)."""
    df = df.copy()
    df["tx_hour"] = (df["TransactionDT"] // 3600) % 24
    df["tx_day"] = (df["TransactionDT"] // 86_400) % 7
    return df


def add_card_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-card daily velocity and ratio of transaction to card historical mean."""
    df = df.copy()
    df["_card_day"] = df["card1"].astype(str) + "_" + df["tx_day"].astype(str)
    df["card_daily_tx_count"] = df.groupby("_card_day")["TransactionAmt"].transform("count")
    card_mean = df.groupby("card1")["TransactionAmt"].transform("mean")
    df["tx_to_card_mean_ratio"] = df["TransactionAmt"] / (card_mean + 1e-9)
    df.drop(columns=["_card_day"], inplace=True)
    return df


def add_email_features(df: pd.DataFrame) -> pd.DataFrame:
    """Flag whether purchaser and recipient share the same email domain."""
    df = df.copy()
    df["same_email_domain"] = (
        df["P_emaildomain"].fillna("") == df["R_emaildomain"].fillna("")
    ).astype(np.int8)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_card_aggregates(df)
    df = add_email_features(df)
    return df
