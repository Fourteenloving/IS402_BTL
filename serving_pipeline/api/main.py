from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import sys
import os

# Ép kiểu UTF-8 cho log Terminal
sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI(title="Customer Churn Prediction API", version="1.0")

# Lấy đường dẫn tuyệt đối tới thư mục 'model' đang nằm cùng chỗ với file main.py này
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(CURRENT_DIR, "model")

# Ép MLflow đọc file vật lý bằng tiền tố file:// (Lưu ý: trên Windows/Docker cần chuyển \ thành /)
model_uri = f"file:///{model_path}".replace("\\", "/")

print(f"Đang tải mô hình từ: {model_uri}...")
try:
    model = mlflow.sklearn.load_model(model_uri)
    print("Đã tải mô hình thành công và sẵn sàng dự đoán!")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    model = None

# Định nghĩa cấu trúc dữ liệu đầu vào (Schema)
class PredictRequest(BaseModel):
    features: dict  # Nhận một dictionary chứa các đặc trưng

# Tạo Endpoint kiểm tra sức khỏe hệ thống (Health check)
@app.get("/")
def read_root():
    return {"message": "API Dự đoán Churn đang hoạt động!"}

# Tạo Endpoint dự đoán
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