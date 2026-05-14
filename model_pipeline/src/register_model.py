import mlflow
from mlflow.tracking import MlflowClient
import json


def register():
    # 1
    with open("model_pipeline/src/run_info.json", "r") as f:
        run_id = json.load(f)["run_id"]

    # 2
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    accuracy = client.get_run(run_id).data.metrics.get("accuracy", 0)

    # 3
    if accuracy >= 0.80:
        model_uri = f"runs:/{run_id}/random_forest_model"
        result = mlflow.register_model(model_uri, "CustomerChurnModel")

        client.transition_model_version_stage(
            name="CustomerChurnModel",
            version=result.version,
            stage="Production",
            archive_existing_versions=True
        )
        print("Done register Production!")


if __name__ == "__main__":
    register()