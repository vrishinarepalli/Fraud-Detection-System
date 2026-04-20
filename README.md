# Fraud Detection System

End-to-end machine learning pipeline for real-time payment fraud detection built on the IEEE-CIS Fraud Detection dataset. The system trains gradient boosted tree models on 590,000+ transactions, exposes a REST API for real-time scoring, and returns a fraud probability with an APPROVED or FLAGGED decision.

---

## Dataset

**Source:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle)  
**Size:** 590,540 training transactions across two CSV files

| File | Rows | Columns | Description |
|---|---|---|---|
| `train_transaction.csv` | 590,540 | 394 | Transaction amounts, card metadata, email domains, engineered V-features |
| `train_identity.csv` | 144,233 | 41 | Device type, browser, anonymized identity features |

The two files join on `TransactionID`. The target column is `isFraud` (1 = fraud, 0 = legitimate). Class balance is approximately 3.5% fraud — a significant imbalance that drives modeling decisions throughout the project.

> **Security:** The dataset CSVs are not tracked in this repository. See `data/README.md` for download instructions. Never commit raw data files, Kaggle API credentials (`kaggle.json`), or any file matching `data/*.csv` to version control.

---

## Project Roadmap

### Phase 1 — Environment Setup
Install dependencies, configure the Python environment, and verify the data loads correctly. Confirm the shape of both CSVs and understand the join cardinality before writing any modeling code.

### Phase 2 — Exploratory Data Analysis
Investigate class imbalance, missing value patterns by column, transaction amount distributions, and time-of-day fraud rates. The goal is to understand what the data contains before making any preprocessing decisions. Output: charts and summary statistics in `notebooks/01_eda.ipynb`.

### Phase 3 — Preprocessing and Feature Engineering
Handle missing values (median fill + binary missing-indicator columns), label-encode categoricals, and build engineered features: transaction hour, day-of-week, per-card daily velocity, ratio of transaction amount to card historical mean, and purchaser/recipient email domain match flag. All fit operations use training data only to prevent data leakage.

### Phase 4 — Modeling
Train three models in sequence using a time-based train/validation split (last 20% of transactions by `TransactionDT` as validation):

1. Logistic regression — establishes a performance floor
2. XGBoost — primary model, industry standard for tabular fraud detection
3. LightGBM — comparison model, faster training with similar accuracy

Class imbalance is addressed via `scale_pos_weight` for tree models and `class_weight='balanced'` for logistic regression. Primary evaluation metric is AUC-ROC; secondary is average precision.

### Phase 5 — Improvement
Hyperparameter search with Optuna (50–100 trials on validation AUC-ROC). SHAP values to validate that the model is learning genuine fraud signals rather than spurious correlations. Threshold optimization via precision-recall curve to find the operating point that best fits the cost structure of false positives versus false negatives. Drift detection via sliding-window KS test on daily score distributions.

### Phase 6 — Deployment
FastAPI application that loads the trained model and preprocessor at startup and exposes two endpoints:

- `GET /health` — liveness probe
- `POST /predict` — accepts a transaction JSON, returns `fraud_probability` and `decision`

Deployment target: Render free tier. The live API URL will be added here once deployed.

### Phase 7 — Documentation
Clean notebooks walking through the full analysis, final results table, and SHAP summary plots committed to the repository.

---

## Tech Stack

| Layer | Libraries |
|---|---|
| Data manipulation | pandas, numpy |
| Modeling | scikit-learn, XGBoost, LightGBM |
| Hyperparameter tuning | Optuna |
| Explainability | SHAP |
| API | FastAPI, uvicorn, pydantic |
| Deployment | Render (free tier) |
| Testing | pytest, httpx |
| Language | Python 3.11 |

---

## Repository Structure

```
fraud-detection-system/
├── .github/
│   └── workflows/
│       └── ci.yml              # Lint, type check, and test on push
├── configs/
│   └── model_config.yaml       # Hyperparameter defaults for all models
├── data/
│   └── README.md               # Dataset download instructions (CSVs are git-ignored)
├── figures/                    # Generated evaluation plots (git-ignored)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_shap_analysis.ipynb
├── artifacts/                  # Saved .pkl model files (git-ignored)
├── scripts/
│   ├── tune_hyperparams.py     # Optuna search
│   └── compute_shap.py         # SHAP summary plot generation
├── src/
│   ├── data/
│   │   ├── loader.py           # CSV loading and train/identity merge
│   │   └── preprocessor.py     # Missing value handling, label encoding
│   ├── features/
│   │   └── engineer.py         # Time features, card velocity, email flags
│   ├── models/
│   │   ├── train.py            # Training pipeline with time-based split
│   │   └── evaluate.py         # Metrics, threshold optimization, plots
│   └── api/
│       ├── main.py             # FastAPI app
│       └── schemas.py          # Pydantic request and response models
├── tests/
│   ├── test_api.py
│   └── test_features.py
├── .gitignore
├── Makefile
├── requirements.txt
├── pyproject.toml
└── SECURITY.md
```

---

## Expected Outcome

A trained XGBoost model with target AUC-ROC of 0.94+ deployed as a REST API on Render. Each POST to `/predict` returns:

```json
{
  "fraud_probability": 0.0312,
  "decision": "APPROVED",
  "threshold_used": 0.5
}
```

| Model | AUC-ROC | Average Precision |
|---|---|---|
| Logistic Regression | TBD | TBD |
| XGBoost | TBD | TBD |
| LightGBM | TBD | TBD |
| XGBoost + Optuna | TBD | TBD |

*Results will be updated after training on the full dataset.*

---

## Security

**Never commit the following to this repository:**

- API keys or tokens of any kind
- `.env` files or any file containing environment variables with credentials
- `kaggle.json` or any Kaggle authentication file
- Trained model `.pkl` files (these may encode training data)
- Raw dataset CSVs from `data/`
- Any file containing passwords, secrets, or private keys

These are enforced via `.gitignore`. If a secret is accidentally committed, treat it as compromised immediately — rotate it, then remove it from git history using `git filter-repo`.

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/vrishinarepalli/fraud-detection-system.git
cd fraud-detection-system

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset — see data/README.md

# 5. Train models
make train

# 6. Run tests
make test

# 7. Start the API server locally
make serve
```

---

## License

MIT
