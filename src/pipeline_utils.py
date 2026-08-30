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

NUMERIC_FEATURES = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL_FEATURES = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "HeartDisease"

ZERO_AS_MISSING = ["RestingBP", "Cholesterol"]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ZERO_AS_MISSING:
        n_zeros = int((df[col] == 0).sum())
        if n_zeros:
            df[col] = df[col].replace(0, np.nan)
    return df


def split_X_y(df: pd.DataFrame, target: str = TARGET):
    X = df[FEATURE_ORDER].copy()
    y = df[target].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
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
    pipe = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
    grid.fit(X_train, y_train)
    return grid


def evaluate_model(grid: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series, name: str = "modelo") -> dict:
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
    y_pred = grid.predict(X_test)
    return confusion_matrix(y_test, y_pred)


def get_classification_report(grid: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series) -> str:
    y_pred = grid.predict(X_test)
    return classification_report(y_test, y_pred)


def predict_single(model, features: list) -> dict:
    """Predice sobre un único registro representado como lista de valores,
    en el orden FEATURE_ORDER. La usa app/api.py en el endpoint /predict.
    """
    row = pd.DataFrame([features], columns=FEATURE_ORDER)
    proba = float(model.predict_proba(row)[0][1])
    return {"heart_disease_probability": proba, "prediction": int(proba > 0.5)}
