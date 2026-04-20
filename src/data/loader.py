from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_raw(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and merge training transaction and identity CSVs."""
    transactions = pd.read_csv(data_dir / "train_transaction.csv")
    identity = pd.read_csv(data_dir / "train_identity.csv")
    return transactions.merge(identity, on="TransactionID", how="left")


def load_test(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and merge test transaction and identity CSVs."""
    transactions = pd.read_csv(data_dir / "test_transaction.csv")
    identity = pd.read_csv(data_dir / "test_identity.csv")
    return transactions.merge(identity, on="TransactionID", how="left")
