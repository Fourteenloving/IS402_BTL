from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

# 1. Trỏ tới file dữ liệu đã qua xử lý của chúng ta
churn_data_source = FileSource(
    path="../../../data_pipeline/data/processed/df_processed.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp"
)

# 2. Định nghĩa Khóa chính (Entity) là CustomerID
customer = Entity(
    name="customer",
    join_keys=["CustomerID"],
    description="ID của khách hàng",
)

# 3. Định nghĩa Feature View (Góc nhìn đặc trưng)
churn_feature_view = FeatureView(
    name="customer_churn_features",
    entities=[customer],
    ttl=timedelta(days=3650),
    source=churn_data_source,
    schema=[
        Field(name="Age", dtype=Int64),
        Field(name="Tenure", dtype=Int64),
        Field(name="Total Spend", dtype=Float32),
        Field(name="Tenure_Age_Ratio", dtype=Float32),
        Field(name="Spend_per_Usage", dtype=Float32),
        Field(name="Churn", dtype=Int64)
    ],
)