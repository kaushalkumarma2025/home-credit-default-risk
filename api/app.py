from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import pandas as pd

from feature_engineering import build_features

# -------------------------------
# CREATE APP FIRST (IMPORTANT)
# -------------------------------
app = FastAPI()

# -------------------------------
# LOAD MODEL
# -------------------------------
with open("models/full_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# -------------------------------
# INPUT SCHEMA
# -------------------------------
class Applicant(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    DAYS_EMPLOYED: float
    DAYS_BIRTH: float = -12000
    CNT_FAM_MEMBERS: float = 1
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float

# -------------------------------
# HOME ROUTE
# -------------------------------
@app.get("/")
def home():
    return {"message": "API running"}

# -------------------------------
# PREDICT ROUTE
# -------------------------------
@app.post("/predict")
def predict(applicant: Applicant):

    data = pd.DataFrame([applicant.dict()])
    data = build_features(data)

    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0

    data = data[feature_columns]

    prob = model.predict_proba(data)[0][1]

    if prob < 0.3:
        risk = "LOW"
    elif prob < 0.6:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    decision = "REJECT" if prob >= 0.50 else "APPROVE"

    # explanation
    reasons = []
    row = data.iloc[0]

    if row.get("CREDIT_TO_INCOME_RATIO", 0) > 0.5:
        reasons.append("High credit relative to income")

    if row.get("EXT_SOURCE_MEAN", 1) < 0.4:
        reasons.append("Low external credit score")

    if not reasons:
        reasons.append("No major risk signals")

    return {
        "default_probability": round(float(prob), 4),
        "decision": decision,
        "risk_level": risk,
        "key_drivers": reasons
    }