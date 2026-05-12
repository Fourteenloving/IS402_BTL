FROM python:3.10-slim

WORKDIR /app

# Thêm dòng này để xem log ngay lập tức
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir fastapi uvicorn pydantic pandas scikit-learn mlflow xgboost

COPY api/ ./api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]