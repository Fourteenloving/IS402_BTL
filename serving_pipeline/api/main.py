from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import sys
import os

# ĐÃ TẮT MONITOR Ở ĐÂY
# from .monitor import router as monitor_router

# Ép kiểu UTF-8 cho log Terminal
sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI(title="Customer Churn Prediction API", version="1.0")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(CURRENT_DIR, "models", "m-5f93d8667ce84c7c824e94181c601b17", "artifacts")
model_uri = f"file:///{model_path}".replace("\\", "/")

print(f"Đang tải mô hình từ: {model_uri}...")
try:
    model = mlflow.sklearn.load_model(model_uri)
    print("Đã tải mô hình thành công và sẵn sàng dự đoán!")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    model = None

class PredictRequest(BaseModel):
    features: dict

@app.get("/")
def read_root():
    return {"message": "API Dự đoán Churn đang hoạt động!"}

@app.post("/predict")
def predict_churn(request: PredictRequest):
    if model is None:
        return {"error": "Mô hình chưa được tải. Vui lòng kiểm tra lại MLflow!"}

    try:
        df_input = pd.DataFrame([request.features])
        prediction = model.predict(df_input)
        result = int(prediction[0])
        return {
            "prediction": result,
            "status": "Khách hàng sắp RỜI BỎ (Churn)" if result == 1 else "Khách hàng AN TOÀN (No Churn)"
        }
    except Exception as e:
        return {"error": f"Lỗi trong quá trình dự đoán: {str(e)}"}

# ĐÃ TẮT MONITOR Ở ĐÂY
# app.include_router(monitor_router)