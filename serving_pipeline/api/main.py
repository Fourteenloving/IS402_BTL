from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from .monitor import generate_drift_artifacts
from .predictor import get_model, persist_predictions, predict_records
from .schemas import BatchPredictionResponse, ChurnInput, HealthResponse, PredictionResponse

app = FastAPI(title="Customer Churn Prediction API", version="2.0.0")


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Customer Churn Prediction API is running"}


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    try:
        get_model()
        return HealthResponse(status="ok", model_ready=True)
    except Exception:
        return HealthResponse(status="error", model_ready=False)


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(payload: ChurnInput, background_tasks: BackgroundTasks) -> PredictionResponse:
    try:
        predictions, logged_rows = predict_records([payload])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    background_tasks.add_task(persist_predictions, logged_rows)
    prediction = predictions[0]
    return PredictionResponse(
        prediction=prediction,
        status="Churn" if prediction == 1 else "No Churn",
    )


@app.post("/batch", response_model=BatchPredictionResponse)
def predict_batch(
    payloads: list[ChurnInput],
    background_tasks: BackgroundTasks,
) -> BatchPredictionResponse:
    if not payloads:
        raise HTTPException(status_code=400, detail="Batch payload is empty")

    try:
        predictions, logged_rows = predict_records(payloads)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    background_tasks.add_task(persist_predictions, logged_rows)
    items = [
        PredictionResponse(prediction=value, status="Churn" if value == 1 else "No Churn")
        for value in predictions
    ]
    return BatchPredictionResponse(total=len(items), predictions=items)


@app.get("/monitor/drift")
def monitor_drift(
    format: Literal["json", "html"] = Query(default="json"),
    reference_path: str | None = Query(default=None),
    current_path: str | None = Query(default=None),
    days: int | None = Query(default=30, ge=1),
    save_html: bool = Query(default=False),
):
    try:
        summary, html_report = generate_drift_artifacts(
            reference_path=reference_path,
            current_path=current_path,
            days=days,
            save_html=save_html or format == "html",
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if format == "html":
        return HTMLResponse(content=html_report)

    return summary
