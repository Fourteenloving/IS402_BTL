FROM python:3.10-slim

WORKDIR /app

# Cài đặt các thư viện cần thiết cho Backend
RUN pip install --no-cache-dir fastapi uvicorn pydantic pandas scikit-learn mlflow xgboost

# Copy toàn bộ thư mục api vào trong container
COPY api/ ./api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]