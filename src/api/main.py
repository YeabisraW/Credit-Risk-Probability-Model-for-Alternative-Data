# src/api/main.py
from api.pydantic_models import CustomerData, RiskResponse
from fastapi import FastAPI
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Credit Risk Prediction API")

# Load trained model
model = joblib.load("best_credit_risk_model.pkl")  # must be in project root

# Root endpoint
@app.get("/")
def root():
    return {"message": "API is running."}

# Predict endpoint
@app.post("/predict", response_model=RiskResponse)
def predict_risk(data: CustomerData):
    features = np.array([list(data.dict().values())])
    risk_prob = model.predict_proba(features)[0][1]
    return RiskResponse(risk_probability=float(risk_prob))
