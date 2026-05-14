import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import json

def evaluate_model():
    # 1
    with open("model_pipeline/src/run_info.json", "r") as f:
        run_id = json.load(f)["run_id"]

    # 2
    data_path = 'data_pipeline/data/processed/df_processed.parquet'
    df = pd.read_parquet(data_path)
    cols_to_drop = ['CustomerID', 'event_timestamp', 'created_timestamp']
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns] + ['Churn'])
    y = df['Churn']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3
    mlflow.set_tracking_uri("http://localhost:5000")
    model = mlflow.sklearn.load_model(f'runs:/{run_id}/random_forest_model')

    # 4
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    # 5
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
    print(f"Done evaluate! Acc: {acc}")

if __name__ == "__main__":
    evaluate_model()