from typing import Optional

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    TransactionAmt: float = Field(..., gt=0, description="Transaction amount in USD")
    ProductCD: str = Field(..., description="Product code: W, H, C, S, or R")
    card1: int
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    TransactionDT: int = Field(..., description="Seconds offset from reference date")

    model_config = {
        "json_schema_extra": {
            "example": {
                "TransactionAmt": 117.5,
                "ProductCD": "W",
                "card1": 13926,
                "card4": "visa",
                "card6": "debit",
                "P_emaildomain": "gmail.com",
                "TransactionDT": 86400,
            }
        }
    }


class PredictionResponse(BaseModel):
    transaction_id: Optional[str] = None
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    decision: str = Field(..., description="APPROVED or FLAGGED")
    threshold_used: float
