# ShopWave churn API in a container (lesson 10.4).
# Build from the course folder (the repo root), not from this folder:
#   docker build -f modules/10-deployment-mlops/04-docker/Dockerfile -t shopwave-churn .
#   docker run --rm -p 8000:8000 shopwave-churn
FROM python:3.12-slim

WORKDIR /app

# 1. Libraries first: this layer is cached until requirements-api.txt changes
COPY modules/10-deployment-mlops/04-docker/requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 2. Our code and the trained model (run train_churn.py before building)
COPY lib/churn_model.py lib/
COPY modules/10-deployment-mlops/02-fastapi/api.py .
COPY models/churn_bundle.joblib models/
ENV CHURN_MODEL_PATH=/app/models/churn_bundle.joblib

# 3. Do not run as root inside the container
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
