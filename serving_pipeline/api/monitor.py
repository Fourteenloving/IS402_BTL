import pandas as pd
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os

router = APIRouter()


@router.get("/monitor/drift", response_class=HTMLResponse)
def get_data_drift_report():
    try:
        # Tạo dữ liệu giả lập (Mock data) để demo cho giảng viên thấy tính năng này hoạt động
        reference_data = pd.DataFrame({
            'Age': [30, 40, 25, 35, 50],
            'Tenure': [12, 24, 6, 18, 36],
            'Total Spend': [500.0, 800.0, 300.0, 600.0, 1000.0]
        })

        current_data = pd.DataFrame({
            'Age': [32, 41, 26, 38, 55],
            'Tenure': [14, 26, 8, 20, 40],
            'Total Spend': [520.0, 810.0, 310.0, 650.0, 1100.0]
        })

        # Tạo báo cáo Data Drift
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(reference_data=reference_data, current_data=current_data)

        # Trả về HTML
        return drift_report.get_html()
    except Exception as e:
        return f"<h1>Lỗi tạo báo cáo Drift: {str(e)}</h1>"