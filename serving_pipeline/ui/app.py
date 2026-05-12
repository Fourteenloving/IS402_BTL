import gradio as gr
import requests

# Địa chỉ API FastAPI của bạn
API_URL = "http://api:8000/predict"

def predict(age, tenure, usage_freq, support_calls, payment_delay, total_spend, last_interaction, gender, sub_type, contract_length):
    # 1. Xử lý logic One-hot encoding để khớp với API
    features = {
        "Age": age,
        "Tenure": tenure,
        "Usage Frequency": usage_freq,
        "Support Calls": support_calls,
        "Payment Delay": payment_delay,
        "Total Spend": total_spend,
        "Last Interaction": last_interaction,
        "Tenure_Age_Ratio": tenure / (age + 1) if age > 0 else 0,
        "Spend_per_Usage": total_spend / (usage_freq + 1) if usage_freq > 0 else 0,
        "Support_Calls_per_Tenure": support_calls / (tenure + 1) if tenure > 0 else 0,

        # Categorical mapping
        "Gender_Male": 1 if gender == "Male" else 0,
        "Subscription Type_Premium": 1 if sub_type == "Premium" else 0,
        "Subscription Type_Standard": 1 if sub_type == "Standard" else 0,
        "Contract Length_Monthly": 1 if contract_length == "Monthly" else 0,
        "Contract Length_Quarterly": 1 if contract_length == "Quarterly" else 0,
    }

    # Tính toán Spending Group
    features["Spending_Group_Medium"] = 1 if 300 <= total_spend < 600 else 0
    features["Spending_Group_High"] = 1 if 600 <= total_spend < 900 else 0
    features["Spending_Group_Very High"] = 1 if total_spend >= 900 else 0

    # Tính toán Tenure Group
    features["Tenure_Group_1-2yr"] = 1 if 12 < tenure <= 24 else 0
    features["Tenure_Group_2-3yr"] = 1 if 24 < tenure <= 36 else 0
    features["Tenure_Group_3+yr"] = 1 if tenure > 36 else 0

    # 2. Gửi request tới API
    try:
        response = requests.post(API_URL, json={"features": features})
        if response.status_code == 200:
            result = response.json()

            # Thêm dòng kiểm tra lỗi này
            if "error" in result:
                return f"Lỗi từ mô hình AI: {result['error']}"

            if result['prediction'] == 1:
                return f"CẢNH BÁO: {result['status']}"
            else:
                return f"AN TOÀN: {result['status']}"
        else:
            return f"Lỗi từ server: {response.text}"
    except Exception as e:
        return f"Lỗi: {e}"
# 3. Thiết kế giao diện Web
with gr.Blocks() as demo:
    gr.Markdown("# Hệ Thống Dự Đoán Khách Hàng Rời Bỏ (Customer Churn)")
    gr.Markdown("Nhập thông tin khách hàng bên dưới để AI phân tích nguy cơ rời bỏ dịch vụ.")

    with gr.Row():
        with gr.Column():
            age = gr.Slider(18, 100, value=30, label="Tuổi (Age)", step=1)
            gender = gr.Radio(["Male", "Female"], label="Giới tính", value="Male")
            tenure = gr.Slider(0, 72, value=12, label="Số tháng sử dụng (Tenure)", step=1)
            sub_type = gr.Dropdown(["Basic", "Standard", "Premium"], label="Loại gói cước", value="Standard")
            contract_length = gr.Dropdown(["Monthly", "Quarterly", "Annual"], label="Thời hạn hợp đồng", value="Monthly")

        with gr.Column():
            usage_freq = gr.Number(value=10, label="Tần suất sử dụng/tháng")
            support_calls = gr.Number(value=2, label="Số cuộc gọi hỗ trợ")
            payment_delay = gr.Number(value=5, label="Số ngày trễ thanh toán")
            total_spend = gr.Number(value=500.0, label="Tổng chi tiêu ($)")
            last_interaction = gr.Number(value=15, label="Số ngày kể từ lần tương tác cuối")

    btn = gr.Button("Dự Đoán", variant="primary")
    output = gr.Textbox(label="Kết quả dự đoán từ AI")

    btn.click(fn=predict, inputs=[age, tenure, usage_freq, support_calls, payment_delay, total_spend, last_interaction, gender, sub_type, contract_length], outputs=output)

if __name__ == "__main__":
    demo.launch(server_port=8501, theme=gr.themes.Soft())
    demo.launch(server_name="0.0.0.0", server_port=8501)