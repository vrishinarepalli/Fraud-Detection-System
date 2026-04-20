from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import PredictionResponse, TransactionRequest
from src.features.engineer import build_features


ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
DECISION_THRESHOLD = 0.5

model_store: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_store["model"] = joblib.load(ARTIFACTS_DIR / "xgboost.pkl")
    model_store["preprocessor"] = joblib.load(ARTIFACTS_DIR / "preprocessor.pkl")
    yield
    model_store.clear()


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time transaction fraud scoring using XGBoost trained on the IEEE-CIS dataset.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": "model" in model_store}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    if "model" not in model_store:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([transaction.model_dump()])
    df = build_features(df)

    feature_cols = [
        c for c in df.columns
        if c not in {"TransactionID", "TransactionDT", "isFraud"}
    ]
    X = model_store["preprocessor"].transform(df[feature_cols])
    fraud_prob = float(model_store["model"].predict_proba(X)[0, 1])
    decision = "FLAGGED" if fraud_prob >= DECISION_THRESHOLD else "APPROVED"

    return PredictionResponse(
        fraud_probability=round(fraud_prob, 4),
        decision=decision,
        threshold_used=DECISION_THRESHOLD,
    )
