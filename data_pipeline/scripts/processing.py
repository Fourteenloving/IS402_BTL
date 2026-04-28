import pandas as pd
import numpy as np
import os


def process_data():
    # 1. Đọc dữ liệu gốc
    file_path = 'C:/Users/HP Victus 16/Downloads/customer_churn_dataset-testing-master.csv'
    print(f"Đang đọc dữ liệu từ: {file_path}")
    df_processed = pd.read_csv(file_path)

    # 2. Xử lý missing values và duplicates [cite: 72-84]
    print(f"Số lượng null ban đầu:\n{df_processed.isnull().sum()}")
    df_processed = df_processed.dropna()

    duplicate_count = df_processed.duplicated(subset=['CustomerID']).sum()
    print(f"Số lượng duplicates theo CustomerID: {duplicate_count}")
    df_processed = df_processed.drop_duplicates(subset=['CustomerID'], keep='first')

    # 3. Convert datatype (Đảm bảo định dạng chuẩn) [cite: 85-92]
    df_processed['CustomerID'] = df_processed['CustomerID'].astype("int64")
    int_columns = ['Age', 'Tenure', 'Support Calls', 'Last Interaction']
    for col in int_columns:
        df_processed[col] = df_processed[col].astype(int)

    # 4. Feature Engineering [cite: 103-117]
    df_processed['Tenure_Age_Ratio'] = df_processed['Tenure'] / (df_processed['Age'] + 1)
    df_processed['Spend_per_Usage'] = df_processed['Total Spend'] / (df_processed['Usage Frequency'] + 1)
    df_processed['Support_Calls_per_Tenure'] = df_processed['Support Calls'] / (df_processed['Tenure'] + 1)

    df_processed['Spending_Group'] = pd.qcut(df_processed['Total Spend'], q=4,
                                             labels=['Low', 'Medium', 'High', 'Very High'])
    df_processed['Tenure_Group'] = pd.cut(df_processed['Tenure'], bins=[0, 12, 24, 36, 100],
                                          labels=['<1yr', '1-2yr', '2-3yr', '3+yr'])

    # 5. One-hot encoding cho các biến Categorical [cite: 94-102, 118-120]
    categorical_features = ['Gender', 'Subscription Type', 'Contract Length', 'Spending_Group', 'Tenure_Group']
    df_processed = pd.get_dummies(df_processed, columns=categorical_features, drop_first=True)

    # Thêm cột timestamp bắt buộc cho Feast
    from datetime import datetime
    df_processed['event_timestamp'] = pd.to_datetime(datetime.now())
    df_processed['created_timestamp'] = pd.to_datetime(datetime.now())

    # 6. Export file đã xử lý [cite: 126-131]
    out_dir = '../data/processed'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'df_processed.csv')
    df_processed.to_csv(out_path, index=False)
    print(f"\nTuyệt vời! Đã xuất dữ liệu sạch ra: {out_path}")


if __name__ == "__main__":
    process_data()