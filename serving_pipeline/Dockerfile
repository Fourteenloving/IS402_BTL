FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir --upgrade pip && \
    pip install -v --no-cache-dir fastapi uvicorn[standard] pandas scikit-learn==1.8.0 mlflow==3.11.1 xgboost evidently

COPY serving_pipeline/api ./api
COPY data_pipeline/data/processed/df_processed.csv ./data_pipeline/data/processed/df_processed.csv

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
