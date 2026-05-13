from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import mlflow.sklearn
import pandas as pd

from .schemas import ChurnInput

API_DIR = Path(__file__).resolve().parent
MODEL_ROOT = API_DIR / "models"
PREDICTIONS_PATH = API_DIR / "reports" / "predictions" / "current_predictions.csv"


def _discover_model_uri() -> str:
    model_files = sorted(MODEL_ROOT.glob("*/artifacts/MLmodel"))
    if not model_files:
        raise FileNotFoundError("No MLflow model artifact was found in serving_pipeline/api/models")
    return model_files[0].parent.resolve().as_uri()


@lru_cache(maxsize=1)
def get_model_uri() -> str:
    return _discover_model_uri()


@lru_cache(maxsize=1)
def get_model():
    return mlflow.sklearn.load_model(get_model_uri())


@lru_cache(maxsize=1)
def get_model_columns() -> tuple[str, ...]:
    model = get_model()
    columns = getattr(model, "feature_names_in_", None)
    if columns is None:
        raise ValueError("Loaded model does not expose feature_names_in_")
    return tuple(columns.tolist())


def _build_feature_row(payload: ChurnInput) -> dict[str, int | float]:
    raw = payload.as_raw_features()
    age = float(raw["Age"])
    tenure = float(raw["Tenure"])
    usage_frequency = float(raw["Usage Frequency"])
    support_calls = float(raw["Support Calls"])
    total_spend = float(raw["Total Spend"])

    return {
        "Age": age,
        "Tenure": tenure,
        "Usage Frequency": usage_frequency,
        "Support Calls": support_calls,
        "Payment Delay": float(raw["Payment Delay"]),
        "Total Spend": total_spend,
        "Last Interaction": float(raw["Last Interaction"]),
        "Tenure_Age_Ratio": tenure / (age + 1.0),
        "Spend_per_Usage": total_spend / (usage_frequency + 1.0),
        "Support_Calls_per_Tenure": support_calls / (tenure + 1.0),
        "Gender_Male": int(raw["Gender"] == "Male"),
        "Subscription Type_Premium": int(raw["Subscription Type"] == "Premium"),
        "Subscription Type_Standard": int(raw["Subscription Type"] == "Standard"),
        "Contract Length_Monthly": int(raw["Contract Length"] == "Monthly"),
        "Contract Length_Quarterly": int(raw["Contract Length"] == "Quarterly"),
        "Spending_Group_Medium": int(300 <= total_spend < 600),
        "Spending_Group_High": int(600 <= total_spend < 900),
        "Spending_Group_Very High": int(total_spend >= 900),
        "Tenure_Group_1-2yr": int(12 < tenure <= 24),
        "Tenure_Group_2-3yr": int(24 < tenure <= 36),
        "Tenure_Group_3+yr": int(tenure > 36),
    }


def _build_inference_frame(payloads: list[ChurnInput]) -> pd.DataFrame:
    rows = [_build_feature_row(payload) for payload in payloads]
    frame = pd.DataFrame(rows)
    return frame.reindex(columns=get_model_columns(), fill_value=0).astype(float)


def predict_records(payloads: list[ChurnInput]) -> tuple[list[int], pd.DataFrame]:
    frame = _build_inference_frame(payloads)
    predictions = [int(value) for value in get_model().predict(frame).tolist()]
    logged_rows = frame.assign(
        prediction=predictions,
        predicted_at=datetime.now(timezone.utc).isoformat(),
    )
    return predictions, logged_rows


def persist_predictions(rows: pd.DataFrame) -> None:
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(PREDICTIONS_PATH, mode="a", header=not PREDICTIONS_PATH.exists(), index=False)


def load_current_predictions(days: int | None = None, current_path: str | None = None) -> pd.DataFrame:
    path = Path(current_path) if current_path else PREDICTIONS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Current prediction log was not found: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("Current prediction log is empty")

    if days and "predicted_at" in frame.columns:
        frame["predicted_at"] = pd.to_datetime(frame["predicted_at"], utc=True, errors="coerce")
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        frame = frame.loc[frame["predicted_at"].ge(cutoff)].copy()

    if frame.empty:
        raise ValueError("No current predictions matched the selected time window")

    return frame
