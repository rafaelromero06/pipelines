"""
api.py
======
API de predicción para el modelo de enfermedad cardíaca (Etapa 3 del
Proyecto Integrador -- sección 10.12).

Carga `app/model.joblib`: el Pipeline COMPLETO (preprocesador + modelo
ganador) exportado en `notebooks/2_model_pipeline_cv.ipynb`. Como el
pipeline incluye imputación, escalado y one-hot encoding, la API recibe
features "crudas" (sin transformar) y deja que el propio pipeline las
procese igual que en entrenamiento.

Nota de diseño: FEATURE_ORDER se duplica aquí (en vez de importarse desde
`src/pipeline_utils.py`) a propósito. El `docker/Dockerfile` solo copia la
carpeta `app/` a la imagen (ver Etapa 3 del enunciado), así que el servicio
de predicción queda intencionalmente desacoplado del código de
entrenamiento/notebooks -- una imagen más liviana y sin dependencias que no
necesita en producción (pandas para servir sí, pero no matplotlib/seaborn).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Mismo orden de columnas usado al entrenar (ver src/pipeline_utils.FEATURE_ORDER)
FEATURE_ORDER = [
    "Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak",
    "Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope",
]

# RestingBP=0 / Cholesterol=0 son fisiológicamente imposibles: en entrenamiento
# se tratan como NaN (ver src/pipeline_utils.load_data). Se replica aquí para
# que la API se comporte igual que el pipeline entrenado (evita train/serve skew).
ZERO_AS_MISSING = ["RestingBP", "Cholesterol"]

# Resuelto relativo a este archivo (no al cwd) para que funcione igual
# corriendo local, en Docker o en Kubernetes.
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
    # Lista posicional de 11 valores, en el orden de FEATURE_ORDER.
    # Ejemplo: [40, 140, 289, 0, 172, 0.0, "M", "ATA", "Normal", "N", "Up"]
    features: list


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
