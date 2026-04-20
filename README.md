# Fraud Detection System

An end-to-end machine learning system for real-time payment fraud detection, built on the IEEE-CIS Fraud Detection dataset. The pipeline covers the full lifecycle: exploratory analysis, feature engineering, model training, hyperparameter optimization, explainability, and production deployment as a REST API.

The trained XGBoost model scores transactions in milliseconds, returning a fraud probability and an APPROVED or FLAGGED decision. The API is deployed on Render and publicly accessible.

---

## Results

| Model | AUC-ROC | Average Precision |
|---|---|---|
| Logistic Regression | TBD | TBD |
| XGBoost | TBD | TBD |
| LightGBM | TBD | TBD |
| XGBoost + Optuna tuning | TBD | TBD |

*Updated after training completes on the full dataset.*

---

## Dataset

**Source:** IEEE-CIS Fraud Detection (Kaggle, 2019)

The dataset contains 590,540 real-world e-commerce transactions split across two files that join on `TransactionID`:

| File | Rows | Columns | Contents |
|---|---|---|---|
| `train_transaction.csv` | 590,540 | 394 | Amounts, card metadata, email domains, engineered V-features |
| `train_identity.csv` | 144,233 | 41 | Device type, browser, anonymized identity attributes |

The fraud rate is approximately 3.5%, making this a heavily imbalanced classification problem. The target column is `isFraud`.

Key columns:

| Column | Description |
|---|---|
| `TransactionDT` | Seconds offset from a reference date (not a Unix timestamp) |
| `TransactionAmt` | Transaction amount in USD |
| `ProductCD` | Product category: W, H, C, S, R |
| `card1`–`card6` | Anonymized card network, type, and issuer metadata |
| `P_emaildomain` / `R_emaildomain` | Purchaser and recipient email domains |
| `C1`–`C14` | Transaction count aggregates per card, address, and email |
| `D1`–`D15` | Days elapsed since prior events (account age, last transaction, etc.) |
| `M1`–`M9` | Match flags: name, address, and card verification matches |
| `V1`–`V339` | Vesta proprietary features (semantics undisclosed) |

---

## How It Was Built

### Exploratory Analysis

Before any modeling, the data was explored to understand class imbalance, missingness patterns, and distributional properties. Several identity columns exceed 90% missingness. Transaction amounts are right-skewed. Fraud rates vary meaningfully by hour of day and by card network. These findings directly informed the preprocessing and feature engineering decisions.

### Preprocessing

The preprocessing pipeline is implemented as a scikit-learn `Pipeline` with two stages:

**Missing value handling:** Numeric columns are filled with the training-set median. For each column with meaningful missingness (>5%), a binary indicator column `{col}_missing` is added. This preserves the signal that data was absent rather than discarding it.

**Categorical encoding:** Low-cardinality columns (`ProductCD`, `card4`, `card6`, email domains, M-flag columns) are label-encoded. At inference time, labels not seen during training are mapped to `"unknown"` rather than raising an error.

All fit operations use only the training split. The preprocessor is serialized alongside the model to ensure consistent transformations at inference time.

### Feature Engineering

Five features were engineered based on domain knowledge of payment fraud:

| Feature | Description |
|---|---|
| `tx_hour` | Hour of day derived from `TransactionDT` |
| `tx_day` | Day of week derived from `TransactionDT` |
| `card_daily_tx_count` | Number of transactions on the same card on the same day |
| `tx_to_card_mean_ratio` | Transaction amount relative to the card's historical average |
| `same_email_domain` | Binary flag: purchaser and recipient share the same email domain |

Velocity features (count, ratio) are computed as group aggregates on the full dataset rather than as rolling windows. This works correctly for the offline training context and is re-computed identically at inference time from the request payload.

### Train / Validation Split

The dataset is split by time: the first 80% of transactions by `TransactionDT` form the training set; the final 20% form the validation set. Random splitting is explicitly avoided because it would leak temporal patterns from the future into training, inflating evaluation metrics in a way that does not reflect production performance.

### Models

Three models are trained in sequence:

1. **Logistic regression** serves as the baseline. It establishes a performance floor and validates that the feature set carries signal before investing in more complex models.

2. **XGBoost** is the primary model. Gradient boosted trees consistently outperform other approaches on tabular fraud data and produce well-calibrated probabilities. Class imbalance is addressed by setting `scale_pos_weight` to the ratio of negative to positive examples (~28:1).

3. **LightGBM** is trained as a comparison. It typically matches XGBoost in accuracy while training faster, making it a useful alternative when iteration speed matters.

All three models are evaluated on AUC-ROC and average precision. Accuracy is not used: a model that predicts no fraud achieves 96.5% accuracy and is operationally worthless.

### Hyperparameter Optimization

Optuna runs a Bayesian search over XGBoost hyperparameters (`max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`, regularization terms) using the validation AUC-ROC as the objective. The best parameters are saved to `artifacts/best_params.yaml` and the final model is retrained with them.

### Explainability

Tree SHAP values are computed on a 5,000-transaction sample of the validation set. The summary plot in `figures/shap_summary.png` shows which features drive predictions and in which direction. This serves two purposes: validating that the model is learning genuine fraud signals rather than spurious correlations, and providing interpretable output for compliance or operations teams.

### Threshold Optimization

The default decision threshold of 0.5 is rarely the optimal operating point for fraud detection. The precision-recall curve is used to find the threshold that maximizes F1, or to select a threshold that satisfies a business constraint (e.g., "catch 80% of fraud"). The chosen threshold is embedded in the API and reported in every response.

### Drift Detection

In production, the distribution of `fraud_probability` scores is logged daily. A sliding-window Kolmogorov-Smirnov test compares recent score distributions to a reference window from shortly after deployment. A significant shift triggers a retraining alert. This approach detects both data drift (changes in transaction patterns) and concept drift (changes in what constitutes fraud).

---

## API

The API is built with FastAPI and served with uvicorn. The model and preprocessor are loaded once at startup and held in memory; there is no disk read per request.

### Endpoints

```
GET  /health    Liveness probe
POST /predict   Score a transaction
```

### Example Request

```bash
curl -X POST https://<render-url>/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 117.5,
    "ProductCD": "W",
    "card1": 13926,
    "card4": "visa",
    "card6": "debit",
    "P_emaildomain": "gmail.com",
    "TransactionDT": 86400
  }'
```

### Example Response

```json
{
  "fraud_probability": 0.0312,
  "decision": "APPROVED",
  "threshold_used": 0.5
}
```

`fraud_probability` is a value between 0 and 1. `decision` is `APPROVED` if the probability is below the threshold, `FLAGGED` otherwise.

---

## Repository Structure

```
fraud-detection-system/
├── .github/workflows/ci.yml   # Lint, type check, and test on every push
├── configs/
│   └── model_config.yaml       # Default hyperparameters for all three models
├── data/
│   └── README.md               # Dataset download instructions
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_shap_analysis.ipynb
├── scripts/
│   ├── tune_hyperparams.py     # Optuna hyperparameter search
│   └── compute_shap.py         # SHAP summary plot generation
├── src/
│   ├── data/
│   │   ├── loader.py           # CSV loading and train/identity merge
│   │   └── preprocessor.py     # Missing value handling and label encoding
│   ├── features/
│   │   └── engineer.py         # Time, velocity, and email domain features
│   ├── models/
│   │   ├── train.py            # Training pipeline with time-based split
│   │   └── evaluate.py         # Metrics, threshold selection, and plots
│   └── api/
│       ├── main.py             # FastAPI application
│       └── schemas.py          # Pydantic request and response models
├── tests/
│   ├── test_api.py
│   └── test_features.py
├── artifacts/                  # Serialized models and preprocessor (git-ignored)
├── figures/                    # Generated evaluation and SHAP plots (git-ignored)
├── .gitignore
├── Makefile
├── requirements.txt
├── pyproject.toml
└── SECURITY.md
```

---

## Tech Stack

| Layer | Libraries |
|---|---|
| Data | pandas, numpy |
| Modeling | scikit-learn, XGBoost, LightGBM |
| Tuning | Optuna |
| Explainability | SHAP |
| API | FastAPI, uvicorn, pydantic |
| Deployment | Render |
| Testing | pytest, httpx |
| Language | Python 3.11 |

---

## Setup

```bash
git clone https://github.com/vrishinarepalli/fraud-detection-system.git
cd fraud-detection-system

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Download the dataset by following the instructions in `data/README.md`, then:

```bash
# Train all three models and save artifacts
make train

# Run tests
make test

# Start the API locally
make serve
```

---

## Security

The following files are excluded from version control via `.gitignore` and must not be committed:

- Dataset CSVs (`data/*.csv`) — too large, and the competition terms restrict redistribution
- Trained model files (`artifacts/*.pkl`) — serialized models can encode training data
- `kaggle.json` — API credential; store it only in `~/.kaggle/` with `chmod 600`
- `.env` files — any environment-specific configuration with secrets

If a credential is accidentally committed, rotate it immediately and remove it from git history using `git filter-repo`. See `SECURITY.md` for the full policy.

---

## License

MIT
