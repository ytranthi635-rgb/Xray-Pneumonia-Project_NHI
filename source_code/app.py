import tensorflow as tf
from tensorflow.keras.models import load_model
import gradio as gr
import numpy as np
import os

print("🚀 Đang tải mô hình MobileNetV2 cho hệ thống X-Ray Pneumonia AI...")
model_path = 'mobilenetv2_final.h5'

if os.path.exists(model_path):
    model = load_model(model_path)
    print("✅ Tải mô hình thành công từ thư mục local!")
else:
    print("⚠️ Không tìm thấy file mô hình mobilenetv2_final.h5! Vui lòng kiểm tra lại thư mục.")

def predict_xray(img):
    if img is None:
        return {"⚠️ Vui lòng tải lên hoặc chụp ảnh X-quang!": 0.0}
    
    # 1. Tiền xử lý ảnh chuẩn theo MobileNetV2 (224x224, chuẩn hóa về [0, 1])
    img = tf.image.resize(img, (224, 224)) / 255.0
    img = tf.expand_dims(img, axis=0)
    
    # 2. Dự đoán xác suất từ mô hình (từ 0.0 đến 1.0)
    pred = float(model.predict(img)[0][0])
    
    # 3. Tính toán tỷ lệ phần trăm chuẩn xác tuyệt đối cho gr.Label
    pneumonia_prob = pred
    normal_prob = 1.0 - pred
    
    return {
        "🔴 Viêm phổi (Pneumonia)": float(pneumonia_prob),
        "🟢 Bình thường (Normal)": float(normal_prob)
    }

# Thiết kế giao diện Gradio với thông điệp cảnh báo y tế nghiêm ngặt và tên tùy chỉnh
css_file = "style.css"

with gr.Blocks(css=css_file if os.path.exists(css_file) else None) as demo:
    
    # Header Top Bar mang tên X-RAY PNEUMONIA AI
    gr.Markdown(
        """
        <div style='display: flex; justify-content: space-between; align-items: center; background: #0f1d40; padding: 14px 22px; border-radius: 12px; border: 1px solid #1e3a8a; margin-bottom: 15px;'>
            <div style='display: flex; align-items: center; gap: 12px;'>
                <span style='font-size: 26px;'>🫁</span>
                <span style='font-size: 20px; font-weight: bold; color: #38bdf8; letter-spacing: 1px;'>X-RAY PNEUMONIA <span style='color: #ffffff;'>AI</span></span>
            </div>
            <div style='display: flex; gap: 20px; font-size: 14px; color: #94a3b8;'>
                <span style='color: #38bdf8; font-weight: 600; cursor: pointer;'>🏠 Tổng quan</span>
                <span style='cursor: pointer;'>📊 Phân tích X-quang</span>
                <span style='cursor: pointer;'>🕒 Lịch sử</span>
                <span style='cursor: pointer;'>📖 Hướng dẫn</span>
            </div>
            <div style='background: #172a54; padding: 6px 12px; border-radius: 8px; font-size: 13px; color: #cbd5e1;'>
                👨‍⚕️ Nghiên cứu sinh ▼
            </div>
        </div>
        
        <div style='background: rgba(239, 68, 68, 0.15); border-left: 4px solid #ef4444; padding: 12px 18px; border-radius: 8px; margin-bottom: 20px;'>
            <p style='color: #fca5a5; font-size: 14px; margin: 0; font-weight: 600; line-height: 1.5;'>
                ⚠️ CẢNH BÁO Y TẾ QUAN TRỌNG: Kết quả từ hệ thống AI này CHỈ MANG TÍNH CHẤT THAM KHẢO VÀ PHỤC VỤ MỤC ĐÍCH NGHIÊN CỨU HỌC TẬP. Mô hình không thể thay thế chẩn đoán lâm sàng của bác sĩ chuyên khoa. Để có kết quả chính xác và an toàn nhất cho sức khỏe, người dùng bắt buộc phải đến các cơ sở y tế uy tín để được bác sĩ trực tiếp thăm khám!
            </p>
        </div>
        """
    )
    
    with gr.Row(equal_height=True):
        # Cột bên trái: Khung tải ảnh hoặc dùng Webcam chụp trực tiếp
        with gr.Column(scale=1):
            gr.Markdown("<h3 style='color: #38bdf8; margin-top: 0;'>🖼️ PHÂN TÍCH ẢNH X-QUANG</h3>")
            image_input = gr.Image(
                sources=["upload", "webcam"], 
                type="numpy", 
                label="Tải ảnh X-quang lên HOẶC dùng Webcam chụp trực tiếp",
                height=350
            )
            with gr.Row():
                clear_btn = gr.Button("🗑️ Xóa ảnh", variant="secondary")
                submit_btn = gr.Button("▶ Phân tích hình ảnh", variant="primary")
                
        # Cột bên phải: Kết quả phân tích AI & Giải thích y tế chi tiết
        with gr.Column(scale=1):
            gr.Markdown("<h3 style='color: #38bdf8; margin-top: 0;'>📊 KẾT QUẢ PHÂN TÍCH AI</h3>")
            label_output = gr.Label(
                num_top_classes=2, 
                label="Tỷ lệ phần trăm dự đoán chi tiết từ mô hình Deep Learning"
            )
            
            gr.Markdown(
                """
                <div style='background: #0f1d40; padding: 15px; border-radius: 12px; border: 1px solid #1e3a8a; margin-top: 15px;'>
                    <h4 style='color: #38bdf8; margin-top: 0;'>💡 GIẢI THÍCH KẾT QUẢ & GIỚI HẠN KHOA HỌC</h4>
                    <p style='font-size: 13px; color: #cbd5e1; line-height: 1.6; margin-bottom: 0;'>
                        • Hệ thống ứng dụng mạng nơ-ron tích chập <b>MobileNetV2</b> đã được huấn luyện trên tập dữ liệu hình ảnh X-quang lồng ngực cộng đồng.<br>
                        • Tỷ lệ phần trăm hiển thị ở trên phản ánh mức độ tương đồng toán học của ảnh đầu vào với các mẫu bệnh lý trong tập dữ liệu huấn luyện, <b>tuyệt đối không phải là kết luận chẩn đoán bệnh lý y khoa cuối cùng</b>.<br>
                        • Sai số hoàn toàn có thể xảy ra do chất lượng ảnh chụp, góc độ ánh sáng hoặc các tổn thương phức tạp khác mà mô hình chưa được học.
                    </p>
                </div>
                
                <div style='background: #172a54; padding: 15px; border-radius: 12px; border: 2px solid #ef4444; margin-top: 15px;'>
                    <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
                        <span style='font-size: 28px;'>🏥</span>
                        <h4 style='color: #fca5a5; margin: 0;'>VAI TRÒ QUYẾT ĐỊNH CỦA BÁC SĨ</h4>
                    </div>
                    <p style='font-size: 13px; color: #f8fafc; line-height: 1.5; margin: 0;'>
                        Bác sĩ chuyên khoa luôn là người giữ vai trò tối quan trọng, kết hợp kết quả chụp chiếu với triệu chứng lâm sàng, xét nghiệm máu và kinh nghiệm thực tế. <b>Không bao giờ được tự ý dựa vào kết quả phần mềm này để tự chẩn đoán hay điều trị tại nhà!</b>
                    </p>
                </div>
                """
            )
            
    # Gắn sự kiện tương tác nút bấm
    submit_btn.click(fn=predict_xray, inputs=image_input, outputs=label_output)
    clear_btn.click(fn=lambda: (None, None), inputs=None, outputs=[image_input, label_output])

if __name__ == "__main__":
    print("🌐 Đang khởi động hệ thống X-Ray Pneumonia AI...")
    demo.launch(share=True)