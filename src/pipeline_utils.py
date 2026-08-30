"""
pipeline_utils.py
==================
Funciones reutilizables de preprocesamiento, entrenamiento y evaluación
para el proyecto "Heart Disease MLOps" (sección 10.12 del curso de
Machine Learning, Prof. Lihki Rubio).

Responde directamente a la Etapa 1, punto 2 del enunciado:
"Separar código en funciones reutilizables: encapsula la lógica de
entrenamiento y evaluación en funciones claras y reutilizables."

Este módulo lo importan ambos notebooks (1_model_leakage_demo.ipynb y
2_model_pipeline_cv.ipynb), la API (app/api.py de forma indirecta, vía
el modelo ya entrenado) y los tests (tests/test_pipeline_utils.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Definición de columnas del dataset "Heart Failure Prediction" (Kaggle)
# --------------------------------------------------------------------------- #
NUMERIC_FEATURES = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL_FEATURES = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "HeartDisease"

# En este dataset, RestingBP=0 y Cholesterol=0 son fisiológicamente imposibles:
# son valores faltantes codificados como 0, no mediciones reales.
ZERO_AS_MISSING = ["RestingBP", "Cholesterol"]


def load_data(path: str) -> pd.DataFrame:
    """Carga el CSV y convierte los ceros imposibles de RestingBP/Cholesterol en NaN.

    Importante: esto NO imputa nada todavía (eso ocurre dentro del pipeline,
    solo con datos de entrenamiento, para evitar data leakage). Solo marca
    los valores especiales como faltantes.
    """
    df = pd.read_csv(path)
    for col in ZERO_AS_MISSING:
        n_zeros = int((df[col] == 0).sum())
        if n_zeros:
            df[col] = df[col].replace(0, np.nan)
    return df


def split_X_y(df: pd.DataFrame, target: str = TARGET):
    """Separa features (en el orden esperado) y variable objetivo."""
    X = df[FEATURE_ORDER].copy()
    y = df[target].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Preprocesador leakage-safe: imputación + escalado/one-hot.

    Se define como parte del Pipeline (no se aplica antes del split), así
    que GridSearchCV lo reajusta en cada fold usando solo el pliegue de
    entrenamiento -- exactamente el patrón que evita el data leakage
    discutido en las secciones 10.1-10.4 del curso.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def train_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model,
    param_grid: dict,
    cv: int = 5,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
) -> GridSearchCV:
    """Entrena Pipeline(preprocesador + modelo) dentro de un GridSearchCV.

    Esta es la función central pedida en la Etapa 1: dado cualquier
    estimador de scikit-learn y su rejilla de hiperparámetros, arma el
    pipeline completo, corre la búsqueda en red con validación cruzada y
    devuelve el GridSearchCV ya ajustado.
    """
    pipe = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
    grid.fit(X_train, y_train)
    return grid


def evaluate_model(grid: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series, name: str = "modelo") -> dict:
    """Evalúa un GridSearchCV ya entrenado sobre el conjunto de prueba.

    Devuelve un diccionario con accuracy, AUC y los mejores hiperparámetros,
    listo para acumular en una tabla comparativa (ranking de modelos).
    """
    y_pred = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:, 1]
    return {
        "modelo": name,
        "mejores_parametros": grid.best_params_,
        "cv_score_roc_auc": round(float(grid.best_score_), 4),
        "accuracy_test": round(float(accuracy_score(y_test, y_pred)), 4),
        "auc_test": round(float(roc_auc_score(y_test, y_proba)), 4),
    }


def get_confusion_matrix(grid: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
    """Matriz de confusión del mejor estimador sobre el conjunto de prueba."""
    y_pred = grid.predict(X_test)
    return confusion_matrix(y_test, y_pred)


def get_classification_report(grid: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series) -> str:
    """Reporte de precisión/recall/F1 por clase, como texto."""
    y_pred = grid.predict(X_test)
    return classification_report(y_test, y_pred)


def predict_single(model, features: list) -> dict:
    """Predice sobre un único registro representado como lista de valores,
    en el orden FEATURE_ORDER. La usa app/api.py en el endpoint /predict.
    """
    row = pd.DataFrame([features], columns=FEATURE_ORDER)
    proba = float(model.predict_proba(row)[0][1])
    return {"heart_disease_probability": proba, "prediction": int(proba > 0.5)}
