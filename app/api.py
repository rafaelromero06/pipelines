from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

FEATURE_ORDER = [
    "Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak",
    "Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope",
]

ZERO_AS_MISSING = ["RestingBP", "Cholesterol"]

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Predice la probabilidad de enfermedad cardíaca a partir de 11 "
        "variables clínicas del dataset Heart Failure Prediction (Kaggle)."
    ),
    version="1.0.0",
)


class InputData(BaseModel):
   
class PredictionResponse(BaseModel):
    prediction: int
    heart_disease_probability: float
    label: str


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Heart Disease Prediction API",
        "expected_features": FEATURE_ORDER,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """Usado por el liveness/readiness probe de Kubernetes (Etapa 4)."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData):
    if len(data.features) != len(FEATURE_ORDER):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Se esperaban {len(FEATURE_ORDER)} valores en el orden "
                f"{FEATURE_ORDER}, se recibieron {len(data.features)}."
            ),
        )

    row = pd.DataFrame([data.features], columns=FEATURE_ORDER)
    for col in ZERO_AS_MISSING:
        row[col] = row[col].replace(0, np.nan)

    try:
        proba = float(model.predict_proba(row)[0][1])
    except Exception as exc:  # tipos/valores inválidos en alguna columna
        raise HTTPException(status_code=400, detail=f"Error al predecir: {exc}") from exc

    pred = int(proba > 0.5)
    return PredictionResponse(
        prediction=pred,
        heart_disease_probability=round(proba, 4),
        label="Con enfermedad" if pred == 1 else "Sin enfermedad",
    )
