import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

def train_model():
    # 1. Đọc dữ liệu từ Feature Store (file Parquet)
    data_path = 'data_pipeline/data/processed/df_processed.parquet'
    print(f"Đang tải dữ liệu từ {data_path}...")
    df = pd.read_parquet(data_path)

    # 2. Loại bỏ các cột không cần thiết cho model
    cols_to_drop = ['CustomerID', 'event_timestamp', 'created_timestamp']
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns] + ['Churn'])
    y = df['Churn']

    # 3. Chia tập huấn luyện (80%) và tập kiểm thử (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Thiết lập MLflow để theo dõi
    mlflow.set_tracking_uri("http://localhost:5000")  # Trỏ tới giao diện MLflow
    mlflow.set_experiment("Customer_Churn_Prediction")

    # Bắt đầu quá trình huấn luyện
    with mlflow.start_run():
        print("Đang huấn luyện mô hình Random Forest...")

        # Tham số của mô hình
        n_estimators = 100
        max_depth = 5
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Dự đoán thử
        y_pred = model.predict(X_test)

        # Đo lường hiệu năng
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        print(f"Hoàn tất! Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

        # Ghi nhận toàn bộ lên MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # Lưu lại file mô hình
        mlflow.sklearn.log_model(model, "random_forest_model")
        print("Đã lưu mô hình vào MLflow Registry thành công!")


if __name__ == "__main__":
    train_model()