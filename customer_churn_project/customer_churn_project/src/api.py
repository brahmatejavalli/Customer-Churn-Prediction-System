"""
api.py
------
FastAPI application for the Customer Churn Prediction System.

Endpoints:
  GET  /              — health check
  GET  /model/info    — model metadata & performance
  POST /predict       — predict churn for one customer
  POST /predict/batch — predict churn for multiple customers
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(__file__))

# ── Load artifacts once at startup ────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')

try:
    model         = joblib.load(os.path.join(MODEL_DIR, 'model.pkl'))
    scaler        = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    MODEL_LOADED  = True
except Exception as e:
    MODEL_LOADED = False
    MODEL_ERROR  = str(e)

# ── FastAPI app ────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predicts the probability of a telecom customer churning "
        "using a Logistic Regression model trained on the Telco dataset."
    ),
    version="1.0.0",
)


# ── Input schema ───────────────────────────────────────────────────
class CustomerData(BaseModel):
    customerID:        Optional[str]   = Field(default="UNKNOWN", example="TC-00123")
    gender:            str             = Field(example="Male")
    SeniorCitizen:     int             = Field(ge=0, le=1, example=0)
    Partner:           str             = Field(example="Yes")
    Dependents:        str             = Field(example="No")
    tenure:            int             = Field(ge=0, example=12)
    PhoneService:      str             = Field(example="Yes")
    MultipleLines:     str             = Field(example="No")
    InternetService:   str             = Field(example="Fiber optic")
    OnlineSecurity:    str             = Field(example="No")
    OnlineBackup:      str             = Field(example="Yes")
    DeviceProtection:  str             = Field(example="No")
    TechSupport:       str             = Field(example="No")
    StreamingTV:       str             = Field(example="Yes")
    StreamingMovies:   str             = Field(example="No")
    Contract:          str             = Field(example="Month-to-month")
    PaperlessBilling:  str             = Field(example="Yes")
    PaymentMethod:     str             = Field(example="Electronic check")
    MonthlyCharges:    float           = Field(ge=0, example=70.35)
    TotalCharges:      str             = Field(example="840.20")

    class Config:
        json_schema_extra = {
            "example": {
                "customerID": "TC-00123",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 5,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": "351.75"
            }
        }


# ── Output schema ──────────────────────────────────────────────────
class PredictionResult(BaseModel):
    customerID:        str
    churn_prediction:  str       # "Yes" or "No"
    churn_probability: float     # 0.0 – 1.0
    risk_level:        str       # Low / Medium / High
    confidence:        str       # human-readable confidence


class BatchPredictionResult(BaseModel):
    total_customers:   int
    churn_count:       int
    churn_rate:        str
    predictions:       List[PredictionResult]


# ── Helper ─────────────────────────────────────────────────────────
def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Fix TotalCharges dtype
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])

    # Feature engineering (must match training)
    df['tenure_group'] = pd.cut(
        df['tenure'], bins=[0, 12, 36, 72],
        labels=['0-1 yr', '1-3 yrs', '3+ yrs'], include_lowest=True
    )
    df['charge_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)

    # Drop customerID — not a feature
    df.drop(columns=['customerID'], inplace=True, errors='ignore')

    # One-hot encode
    df = pd.get_dummies(df, drop_first=True)

    # Align to training columns
    df = df.reindex(columns=feature_names, fill_value=0)

    # Scale numeric features
    numeric_cols = [c for c in ['tenure', 'MonthlyCharges', 'TotalCharges', 'charge_ratio']
                    if c in df.columns]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df


def get_risk_level(prob: float) -> str:
    if prob < 0.30:   return "Low"
    if prob < 0.60:   return "Medium"
    return "High"


def get_confidence(prob: float) -> str:
    p = max(prob, 1 - prob)
    if p >= 0.80:  return f"High confidence ({p:.0%})"
    if p >= 0.60:  return f"Moderate confidence ({p:.0%})"
    return f"Low confidence ({p:.0%})"


def build_result(customer_id: str, prob: float) -> PredictionResult:
    return PredictionResult(
        customerID        = customer_id,
        churn_prediction  = "Yes" if prob >= 0.5 else "No",
        churn_probability = round(float(prob), 4),
        risk_level        = get_risk_level(prob),
        confidence        = get_confidence(prob),
    )


# ── Routes ─────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "message": "Customer Churn Prediction API is running."
    }


@app.get("/model/info", tags=["Model"])
def model_info():
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "model_type":    type(model).__name__,
        "n_features":    len(feature_names),
        "feature_names": feature_names,
        "performance": {
            "accuracy":  "80.1%",
            "roc_auc":   "0.760",
            "cv_roc_auc": "0.768 ± 0.009",
        },
        "model_selection": {
            "candidates": [
                {"model": "Logistic Regression",  "roc_auc": 0.760, "accuracy": "80.1%", "selected": True},
                {"model": "Gradient Boosting",    "roc_auc": 0.757, "accuracy": "78.9%", "selected": False},
                {"model": "Random Forest",        "roc_auc": 0.716, "accuracy": "77.8%", "selected": False},
            ],
            "selection_reason": (
                "Logistic Regression achieved highest ROC-AUC (0.760) with best "
                "cross-validation stability (std=0.009). Preferred for interpretability "
                "and lower risk of overfitting on this dataset size."
            )
        }
    }


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict_single(customer: CustomerData):
    """
    Predict churn probability for a single customer.

    Returns churn prediction (Yes/No), probability score (0–1),
    risk level (Low/Medium/High), and a confidence descriptor.
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        data = customer.dict()
        customer_id = data.get("customerID", "UNKNOWN")

        df_input = preprocess_input(data)
        prob = model.predict_proba(df_input)[0][1]

        return build_result(customer_id, prob)

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResult, tags=["Prediction"])
def predict_batch(customers: List[CustomerData]):
    """
    Predict churn for a list of customers in one request.

    Returns individual predictions plus aggregate statistics
    (total churn count and churn rate across the batch).
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if len(customers) > 500:
        raise HTTPException(status_code=400, detail="Batch size limit is 500 customers.")

    try:
        results = []
        for customer in customers:
            data = customer.dict()
            customer_id = data.get("customerID", "UNKNOWN")
            df_input = preprocess_input(data)
            prob = model.predict_proba(df_input)[0][1]
            results.append(build_result(customer_id, prob))

        churn_count = sum(1 for r in results if r.churn_prediction == "Yes")

        return BatchPredictionResult(
            total_customers = len(results),
            churn_count     = churn_count,
            churn_rate      = f"{churn_count / len(results):.1%}",
            predictions     = results
        )

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Batch prediction error: {str(e)}")
