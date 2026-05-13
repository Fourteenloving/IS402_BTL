FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# 1. Thêm dòng này để nâng cấp công cụ tải gói lên bản mới nhất
RUN pip install --upgrade pip

# 2. Lệnh cài đặt
RUN pip install --no-cache-dir fastapi uvicorn pydantic pandas scikit-learn==1.8.0 mlflow xgboost evidently

COPY api/ ./api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]