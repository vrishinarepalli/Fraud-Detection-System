# Data

This directory holds the raw IEEE-CIS Fraud Detection dataset. The CSV files are not tracked in git due to size (~500 MB uncompressed).

## Download Instructions

1. Create a Kaggle account at https://www.kaggle.com
2. Accept the competition rules at https://www.kaggle.com/c/ieee-fraud-detection
3. Install the Kaggle CLI:
   ```bash
   pip install kaggle
   ```
4. Download your API token from https://www.kaggle.com/settings (Account > API > Create New Token).
   This saves a file called `kaggle.json`.

   **Important:** Place `kaggle.json` in `~/.kaggle/kaggle.json` and set permissions:
   ```bash
   mkdir -p ~/.kaggle
   mv kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
   Never move `kaggle.json` into this repository directory. It is a credential file.

5. Download and extract the data:
   ```bash
   kaggle competitions download -c ieee-fraud-detection -p data/
   cd data && unzip ieee-fraud-detection.zip && rm ieee-fraud-detection.zip
   ```

## Expected Files After Extraction

```
data/
├── train_transaction.csv   # 590,540 rows, 394 columns
├── train_identity.csv      # 144,233 rows, 41 columns
├── test_transaction.csv    # 506,691 rows, 393 columns (no isFraud column)
└── test_identity.csv       # 141,907 rows, 41 columns
```

## Key Columns

**train_transaction.csv**

| Column | Description |
|---|---|
| `TransactionID` | Join key with identity table |
| `TransactionDT` | Seconds offset from an undisclosed reference date (not a Unix timestamp) |
| `TransactionAmt` | Transaction amount in USD |
| `ProductCD` | Product type: W, H, C, S, R |
| `card1`–`card6` | Anonymized card metadata (network, type, etc.) |
| `P_emaildomain` | Purchaser email domain |
| `R_emaildomain` | Recipient email domain |
| `C1`–`C14` | Count features (transaction counts per entity) |
| `D1`–`D15` | Timedelta features (days since a prior event) |
| `M1`–`M9` | Match flags (name, address, etc.) |
| `V1`–`V339` | Vesta-engineered features (semantics undisclosed) |
| `isFraud` | Target: 1 = fraud, 0 = legitimate |

**train_identity.csv**

| Column | Description |
|---|---|
| `TransactionID` | Join key |
| `DeviceType` | `mobile` or `desktop` |
| `DeviceInfo` | Device name and OS string |
| `id_01`–`id_38` | Anonymized identity features |
