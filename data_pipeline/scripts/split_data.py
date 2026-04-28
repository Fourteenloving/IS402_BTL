import pandas as pd
import numpy as np
import os

def split_data():
    # 1. Đọc dữ liệu gốc
    file_path = 'C:/Users/HP Victus 16/Downloads/customer_churn_dataset-testing-master.csv'
    print(f"Đang đọc dữ liệu từ: {file_path}")
    df_train = pd.read_csv(file_path)

    # 2. Chia thành 10 phần [cite: 58]
    splits = np.array_split(df_train, 10)

    # 3. Xuất ra thư mục raw [cite: 62, 63]
    out_dir = '../data/raw'
    for i, df in enumerate(splits, 1):
        out_path = os.path.join(out_dir, f'train_period_{i}.csv')
        df.to_csv(out_path, index=False)
        print(f"Đã lưu: {out_path}")

if __name__ == "__main__":
    split_data()