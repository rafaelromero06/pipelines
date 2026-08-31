# Heart Disease MLOps



**Dataset**: [Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
(Kaggle, `fedesoriano`) — 918 pacientes, 11 variables clínicas, variable
objetivo binaria `HeartDisease` (1 = presenta enfermedad cardíaca).

## Estructura del proyecto

```
heart-disease-mlops/
├── app/
│   ├── api.py              # API FastAPI (Etapa 3)
│   └── model.joblib         # Pipeline entrenado (preprocesador + modelo), exportado en el Notebook 2
├── data/
│   └── heart.csv
├── docker/
│   ├── Dockerfile
│   └── requirements.txt     # dependencias mínimas de producción
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── monitoring/
│   └── generate_drift_report.py   # genera drift_report.html (Etapa 6)
├── notebooks/
│   ├── 1_model_leakage_demo.ipynb    # Etapa 1
│   └── 2_model_pipeline_cv.ipynb     # Etapa 2
├── src/
│   └── pipeline_utils.py    # funciones reutilizables de entrenamiento/evaluación
├── tests/
│   ├── test_api.py
│   └── test_pipeline_utils.py
├── .github/workflows/
│   └── ci.yml                # Etapa 5
├── .flake8
├── conftest.py
├── drift_report.html         # generado por monitoring/generate_drift_report.py
├── requirements.txt           # dependencias de desarrollo (notebooks, tests)
└── README.md
```

## 1. Preparar el entorno

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Etapas 1 y 2 — Notebooks

```bash
jupyter lab notebooks/
```

- **`1_model_leakage_demo.ipynb`**: demuestra el efecto de la fuga de datos
  (data leakage) y compara 5 modelos (`SVC`, `LogisticRegression`,
  `RandomForest`, `KNN`, `GradientBoosting`) con `Pipeline` + `GridSearchCV`.
- **`2_model_pipeline_cv.ipynb`**: split seguro, selección del mejor modelo
  entre los 5 candidatos por AUC de validación cruzada, evaluación (matriz de
  confusión, curva ROC, AUC) y exportación a `app/model.joblib`.

**Resultado real**: `RandomForestClassifier` gana la
selección (`max_depth=10`, `n_estimators=100`) con AUC de validación cruzada
≈ **0.924** y AUC en test ≈ **0.932**. Si vuelves a ejecutar los notebooks
(sobre todo si cambias `random_state` o los datos), estos números pueden
variar ligeramente.



## 3. Etapa 3 — API local

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

Documentación interactiva en `http://localhost:8000/docs`. Ejemplo de uso:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [40, 140, 289, 0, 172, 0.0, "M", "ATA", "Normal", "N", "Up"]}'
```

El orden de `features` es siempre:
`Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak, Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope`
(el mismo orden que devuelve `GET /`).

## 4. Etapa 3 — Docker

```bash
docker build -t heart-api -f docker/Dockerfile .
docker run -p 8000:8000 heart-api
```

## 5. Etapa 4 — Kubernetes (local, con minikube)

```bash
minikube start
eval $(minikube docker-env)              # que minikube use la imagen local
docker build -t heart-api:latest -f docker/Dockerfile .

kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

kubectl get pods
kubectl get svc
minikube service heart-api-service --url
```

## 6. Etapa 5 — Lint y tests (local, igual a lo que corre en CI)

```bash
flake8 .
pytest tests/ -v
```

`.github/workflows/ci.yml` corre ambos pasos automáticamente en cada `push`
o `pull_request` a `main`. Para que el CI pase, `app/model.joblib` y
`data/heart.csv` deben estar **comprometidos al repositorio** (el workflow
no vuelve a entrenar el modelo, solo lintea y prueba lo ya exportado).

## 7. Etapa 6 — Monitoreo de data drift

```bash
python monitoring/generate_drift_report.py
```

Genera `drift_report.html` en la raíz del proyecto (ábrelo en el navegador).
Compara una partición "reference" (datos de entrenamiento) contra una
partición "current" simulada -- en producción, "current" vendría de las
peticiones reales que recibe la API.


