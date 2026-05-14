import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import json

def train_model():
    # 1
    data_path = 'data_pipeline/data/processed/df_processed.parquet'
    df = pd.read_parquet(data_path)

    # 2
    cols_to_drop = ['CustomerID', 'event_timestamp', 'created_timestamp']
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns] + ['Churn'])
    y = df['Churn']

    # 3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Customer_Churn_Prediction")

    with mlflow.start_run() as run:
        n_estimators = 100
        max_depth = 5
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.sklearn.log_model(model, "random_forest_model")

        # 5
        run_id = run.info.run_id
        with open("model_pipeline/src/run_info.json", "w") as f:
            json.dump({"run_id": run_id}, f)
        print("Done train!")

if __name__ == "__main__":
    train_model()