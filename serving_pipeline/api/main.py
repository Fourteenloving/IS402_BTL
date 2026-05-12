from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import sys

# Ép kiểu UTF-8 cho log Terminal
sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI(title="Customer Churn Prediction API", version="1.0")

# 1. Trỏ tới MLflow Server đang chạy
mlflow.set_tracking_uri("http://localhost:5000")

# 2. Thay Run ID của bạn vào đây (Lấy từ log bước trước)
RUN_ID = "e8bb2ecb4a2d44e08ed839e02fb08688"
model_uri = "mlartifacts/1/e8bb2ecb4a2d44e08ed839e02fb08688/artifacts/random_forest_model"

# Tải mô hình lên RAM khi API vừa khởi động
print(f"Đang tải mô hình từ MLflow (Run ID: {RUN_ID})...")
try:
    model = mlflow.sklearn.load_model(model_uri)
    print("Đã tải mô hình thành công và sẵn sàng dự đoán!")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    model = None


# 3. Định nghĩa cấu trúc dữ liệu đầu vào (Schema)
class PredictRequest(BaseModel):
    features: dict  # Nhận một dictionary chứa các đặc trưng


# 4. Tạo Endpoint kiểm tra sức khỏe hệ thống (Health check)
@app.get("/")
def read_root():
    return {"message": "API Dự đoán Churn đang hoạt động!"}


# 5. Tạo Endpoint dự đoán
@app.post("/predict")
def predict_churn(request: PredictRequest):
    if model is None:
        return {"error": "Mô hình chưa được tải. Vui lòng kiểm tra lại MLflow!"}

    try:
        # Chuyển dữ liệu JSON thành DataFrame
        df_input = pd.DataFrame([request.features])

        # Thực hiện dự đoán
        prediction = model.predict(df_input)

        # Trả về kết quả
        result = int(prediction[0])
        return {
            "prediction": result,
            "status": "Khách hàng sắp RỜI BỎ (Churn)" if result == 1 else "Khách hàng AN TOÀN (No Churn)"
        }
    except Exception as e:
        return {"error": f"Lỗi trong quá trình dự đoán: {str(e)}"}