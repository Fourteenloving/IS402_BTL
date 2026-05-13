from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import ClassificationPreset, DataDriftPreset

from .predictor import get_model_columns, load_current_predictions

PROJECT_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents if (parent / "data_pipeline").exists()),
    Path(__file__).resolve().parents[1],
)
DEFAULT_REFERENCE_PATH = PROJECT_ROOT / "data_pipeline" / "data" / "processed" / "df_processed.csv"
HTML_REPORT_DIR = Path(__file__).resolve().parent / "reports" / "drift"


def _load_reference_data(reference_path: str | None = None) -> pd.DataFrame:
    path = Path(reference_path) if reference_path else DEFAULT_REFERENCE_PATH
    if not path.exists():
        raise FileNotFoundError(f"Reference dataset was not found: {path}")
    return pd.read_csv(path)


def _select_common_columns(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    feature_columns = [column for column in get_model_columns() if column in reference_df.columns and column in current_df.columns]
    if not feature_columns:
        raise ValueError("Reference and current datasets do not share model feature columns")

    reference_data = reference_df[feature_columns].copy()
    current_data = current_df[feature_columns].copy()
    include_classification = {"Churn", "prediction"}.issubset(reference_df.columns) and {"Churn", "prediction"}.issubset(current_df.columns)

    if include_classification:
        reference_data["Churn"] = reference_df["Churn"]
        reference_data["prediction"] = reference_df["prediction"]
        current_data["Churn"] = current_df["Churn"]
        current_data["prediction"] = current_df["prediction"]

    return reference_data.astype(float), current_data.astype(float), include_classification


def _build_report(reference_data: pd.DataFrame, current_data: pd.DataFrame, include_classification: bool):
    metrics = [DataDriftPreset()]
    if include_classification:
        metrics.append(ClassificationPreset())
    return Report(metrics).run(reference_data=reference_data, current_data=current_data)


def _summarize_metrics(report_dict: dict, reference_rows: int, current_rows: int, total_features: int) -> dict:
    drifted_features = 0
    drift_share = 0.0
    feature_drift: dict[str, dict[str, float | bool]] = {}

    for metric in report_dict.get("metrics", []):
        metric_name = metric.get("metric_name", "")
        config = metric.get("config", {})
        value = metric.get("value")

        if metric_name.startswith("DriftedColumnsCount"):
            drifted_features = int(value.get("count", 0))
            drift_share = float(value.get("share", 0.0))
            continue

        column = config.get("column")
        threshold = float(config.get("threshold", 0.05))
        if column and isinstance(value, (int, float)):
            p_value = float(value)
            feature_drift[column] = {
                "p_value": p_value,
                "drift_detected": p_value < threshold,
            }

    if drift_share > 0.5:
        drift_status = "high"
    elif drift_share > 0.2:
        drift_status = "medium"
    else:
        drift_status = "low"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_rows": reference_rows,
        "current_rows": current_rows,
        "dataset_drift": drift_share > 0.5,
        "drift_score": drift_share,
        "drift_status": drift_status,
        "drifted_features": drifted_features,
        "total_features": total_features,
        "feature_drift": feature_drift,
    }


def generate_drift_artifacts(
    reference_path: str | None = None,
    current_path: str | None = None,
    days: int | None = None,
    save_html: bool = False,
) -> tuple[dict, str]:
    reference_df = _load_reference_data(reference_path)
    current_df = load_current_predictions(days=days, current_path=current_path)
    reference_data, current_data, include_classification = _select_common_columns(reference_df, current_df)

    snapshot = _build_report(reference_data, current_data, include_classification)
    total_features = len(reference_data.columns) - (2 if include_classification else 0)
    summary = _summarize_metrics(snapshot.dict(), len(reference_data), len(current_data), total_features)
    html_report = ""

    if save_html:
        html_report = snapshot.get_html_str(False)
        HTML_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = HTML_REPORT_DIR / f"drift_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.html"
        report_path.write_text(html_report, encoding="utf-8")
        summary["html_report_path"] = str(report_path)

    return summary, html_report
