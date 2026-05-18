"""
Module for FastAPI app
"""

import logging
import os
import uvicorn
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from src.ModelPredictions import model_predictions_pipeline

app = FastAPI()


# Pydantic model for the input data
class LoanApplication(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: int
    loan_amount: int
    loan_term: int
    cibil_score: int
    residential_assets_value: int
    commercial_assets_value: int
    luxury_assets_value: int
    bank_asset_value: int


@app.post("/predict")
def predict(loan: LoanApplication):
    df = pd.DataFrame([loan.model_dump()])

    model = model_predictions_pipeline(df)

    return {"predictions": model.predictions_labels.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
