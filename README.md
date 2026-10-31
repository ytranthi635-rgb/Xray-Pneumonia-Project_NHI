# Xray-Pneumonia-Project_NHI
Bài tập lớn Học máy: Phân loại viêm phổi qua X-quang với MobileNetV2 và Giao diện Web.
# 🫁 X-RAY PNEUMONIA AI — HỆ THỐNG PHÂN LOẠI VÀ CHẨN ĐOÁN VIÊM PHỔI TỪ ẢNH X-QUANG

Bài tập lớn môn **Học máy (Machine Learning / Deep Learning)**.

---

## 📋 GIỚI THIỆU ĐỀ TÀI
Ứng dụng Trí tuệ nhân tạo (AI) nhằm hỗ trợ tự động phát hiện và phân loại bệnh viêm phổi (Pneumonia) từ ảnh chụp X-quang lồng ngực. Hệ thống kết hợp giữa mô hình học sâu **MobileNetV2** và mô hình Baseline **SVM + HOG**, đi kèm giao diện web trực quan hỗ trợ cả tải ảnh lẫn chụp qua Webcam.

---

## 🛠️ CẤU TRÚC THƯ MỤC DỰ ÁN
```text
Pneumonia-Project/
│
├── report/                  <-- Chứa tài liệu báo cáo bài tập lớn
├── source_code/             <-- Thư mục mã nguồn chính của ứng dụng
│   ├── app.py               <-- Giao diện Web Gradio (Hỗ trợ Upload & Webcam)
│   ├── style.css            <-- Tệp tùy chỉnh giao diện (Dark Mode y tế cao cấp)
│   ├── mobilenetv2_final.h5 <-- Tệp trọng số mô hình Deep Learning đã huấn luyện
│   └── chay_web.bat         <-- Tệp khởi động nhanh hệ thống
│
├── requirements.txt         <-- Danh sách các thư viện cần thiết
└── README.md                <-- Hướng dẫn sử dụng dự án
