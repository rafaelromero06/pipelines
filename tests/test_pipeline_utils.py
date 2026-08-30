"""Tests de src/pipeline_utils.py -- corridos por pytest en .github/workflows/ci.yml (Etapa 5)."""
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.pipeline_utils import (
    FEATURE_ORDER,
    TARGET,
    build_preprocessor,
    evaluate_model,
    load_data,
    split_X_y,
    train_pipeline,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "heart.csv"


def test_load_data_marks_zero_as_missing():
    df = load_data(str(DATA_PATH))
    assert df["Cholesterol"].isna().sum() > 0


def test_split_X_y_shapes():
    df = load_data(str(DATA_PATH))
    X, y = split_X_y(df)
    assert list(X.columns) == FEATURE_ORDER
    assert len(X) == len(y) == len(df)
    assert TARGET not in X.columns


def test_build_preprocessor_fits_without_error():
    df = load_data(str(DATA_PATH))
    X, _ = split_X_y(df)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == X.shape[0]


def test_train_pipeline_and_evaluate_model():
    df = load_data(str(DATA_PATH))
    X, y = split_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    grid = train_pipeline(
        X_train, y_train,
        LogisticRegression(max_iter=1000),
        {"model__C": [1.0]},
        cv=3,
    )
    result = evaluate_model(grid, X_test, y_test, name="test-model")
    assert 0.0 <= result["auc_test"] <= 1.0
    assert 0.0 <= result["accuracy_test"] <= 1.0
