# docker/Dockerfile
# Build:  docker build -t heart-api -f docker/Dockerfile .
# Run:    docker run -p 8000:8000 heart-api
# (el build context es la raíz del repo -- por eso las rutas de COPY son
#  "app/..." y "docker/requirements.txt", no rutas relativas a docker/)

FROM python:3.12-slim

WORKDIR /code

# 1) Dependencias primero -> se cachea esta capa mientras no cambie requirements.txt
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Código de la API + modelo entrenado (app/model.joblib)
COPY app/ ./app/

EXPOSE 8000

# Usado por Docker (y equivalente al probe de Kubernetes en k8s/deployment.yaml)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
