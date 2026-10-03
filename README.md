# 🏙️ AI Định giá Bất động sản Hà Nội (Hanoi Room Pricing AI)

Dự án Data Science & Machine Learning chuyên sâu ứng dụng kiến trúc **Hybrid Detrending (Linear Regression + HistGradientBoosting)** và **BallTree Spatial Indexing** để định giá thuê phòng trọ/chung cư mini tại Hà Nội. Tích hợp giao diện **FastAPI + Leaflet + TailwindCSS Dark Mode**.

---

## 📂 Cấu trúc Dự án (5 Giai đoạn)

Dự án được tổ chức theo tiêu chuẩn vòng đời của một hệ thống Data Science (End-to-End Pipeline):

*   **`1_Data_Collection_Cleaning/`**: Khai phá, cào dữ liệu, làm sạch, và định vị tọa độ vệ tinh qua OpenStreetMap (Geocoding).
*   **`2_Deep_EDA/`**: Phân tích dữ liệu thám hiểm (EDA), biểu diễn trực quan các mẫu phân phối giá và mật độ bất động sản.
*   **`3_Feature_Engineering/`**: Trích xuất đặc trưng NLP, xây dựng tính năng không gian (Spatial Features) bằng Haversine.
*   **`4_Model_Training/`**: Huấn luyện mô hình, kiểm soát Data Leakage, đóng gói Pipeline hoàn chỉnh (`.joblib`).
*   **`5_Web_Deployment/`**: Triển khai ứng dụng Web tương tác với FastAPI và bản đồ động Leaflet.
*   **`data/`**: Chứa toàn bộ dữ liệu thô và dữ liệu đã qua xử lý.
*   **`models/`**: Chứa mô hình AI đã đóng gói phục vụ suy luận.
*   **`static/`**: Tài nguyên Frontend (HTML, CSS, JS).

---

## 🚀 Hướng dẫn Khởi chạy (How to Run)

### 1. Cài đặt môi trường
Vui lòng cài đặt các thư viện bắt buộc thông qua `pip`:
```bash
pip install -r requirements.txt
```

### 2. Khởi chạy Ứng dụng Web
Khởi động máy chủ FastAPI từ thư mục gốc của dự án:
```bash
python 5_Web_Deployment/api.py
```
Sau đó, mở trình duyệt và truy cập: **`http://localhost:8000`**

### 3. Huấn luyện lại mô hình (Re-train)
Nếu bạn có tập dữ liệu mới và muốn train lại mô hình từ đầu (chạy toàn bộ Data Cleaning -> Spatial BallTree -> Detrending -> HistGBM):
```bash
cd 4_Model_Training
python pipeline_production.py --data ../data/phongtro_hanoi_advanced_nlp.csv --geo ../data/geocoded_addresses_checkpoint.csv --output ../models/pipeline_production.joblib
```

---

## 💡 Điểm nổi bật Kỹ thuật (Tech Stack & Highlights)

*   **Machine Learning**: `scikit-learn` (HistGradientBoostingRegressor, OrdinalEncoder, BallTree). Khắc phục độ chệch lạm phát thời gian bằng thuật toán Detrending.
*   **Geospatial Analysis**: `geopy` (Nominatim API), Fallback nhiều tầng chống lỗi từ OpenStreetMap.
*   **Backend & Frontend**: `FastAPI` (Python), `Leaflet.js` (Bản đồ tương tác không lưu cache), `Tailwind CSS` (Giao diện Dark Mode lấy cảm hứng từ XenForo).
*   **Explainable AI**: Giao diện cung cấp hộp thoại phân tích lý do tăng/giảm giá ("Phân tích tác động") dưa trên trọng số đặc trưng của mô hình.
