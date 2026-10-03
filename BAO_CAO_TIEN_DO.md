# Báo cáo tiến độ cải thiện dự án

**Ngày cập nhật:** 03/10/2026  
**Dự án:** Phân tích và dự đoán giá phòng trọ Hà Nội  
**Phạm vi báo cáo:** Kiểm tra chất lượng dữ liệu, sửa pipeline, tái tạo dữ liệu và huấn luyện model production.

## 1. Tóm tắt

Đợt cải thiện này tập trung vào tính đúng đắn của feature thời gian và không gian, quy tắc khử trùng, khả năng tái lập môi trường và kiểm chứng model. Các sửa đổi đã được áp dụng; dữ liệu feature được tạo lại; hai test hồi quy, phân tích feature importance và huấn luyện production đều chạy thành công.

Đã phát hiện quy tắc khử trùng production cũ loại nhầm 115 tin đăng có cùng địa chỉ, giá và ngày nhưng khác diện tích. Quy tắc mới giữ các tin khác diện tích/quận và thống nhất khóa khử trùng với bước chuẩn bị dữ liệu.

## 2. Các hạng mục đã hoàn thành

### 2.1. Feature thị trường theo thời gian

- Chỉ dùng các tin có thời gian hợp lệ và đăng trước tin đang xét.
- Xử lý các tin có cùng timestamp theo một nhóm, tránh để tin cùng thời điểm dùng giá của nhau.
- Tin thiếu thời điểm đăng không được xem là dữ liệu lịch sử.
- Các feature thị trường được tạo lại trong `data/phongtro_hanoi_features.csv`.

### 2.2. Feature không gian và ghép geocode

- Không gán tọa độ median cho tin thiếu geocode.
- Tin không có tọa độ hợp lệ nhận `NaN` cho cả median giá hàng xóm lẫn số lượng hàng xóm; các tin này cũng không thể trở thành hàng xóm của tin khác.
- Khi cache geocode có nhiều dòng cho cùng địa chỉ, ưu tiên dòng có đủ lat/lon.
- Khử trùng được sắp xếp ổn định theo thời gian; timestamp thiếu được đưa xuống cuối.

### 2.3. Quy tắc khử trùng

- Khóa production đã thống nhất với bước chuẩn bị dữ liệu: `exact_address`, `area`, `price`, `district`.
- Audit trên dữ liệu sau lọc domain cho thấy khóa cũ giữ 6.251 dòng; khóa mới giữ 6.366 dòng.
- 115 dòng được giữ lại không phải các bản sao tương đương: chúng thuộc 29 nhóm có cùng địa chỉ/giá/ngày theo khóa cũ; mọi nhóm đều có khác biệt diện tích, 25 nhóm có khác biệt quận, và các dòng có `post_id` riêng.

### 2.4. NLP, missing values và gọi geocoding

- Sửa nhận diện phủ định trong feature NLP, bao gồm cụm “không chung chủ”.
- Giữ giá trị tiện ích chưa được đề cập ở dạng missing thay vì tự gán là không có.
- Thêm khoảng nghỉ sau mọi request Nominatim, kể cả khi geocode thành công.

### 2.5. Đánh giá model và môi trường

- Feature importance chia train/test theo thời gian và nhận diện đúng tên feature.
- Chuyển permutation importance sang chạy tuần tự để tránh lỗi khởi tạo worker process trên Windows.
- Thêm `requirements.txt` với các phiên bản thư viện đã dùng để kiểm tra.

## 3. Tình trạng dữ liệu

Các con số dưới đây thuộc hai nhánh dữ liệu riêng của dự án:

- Nhánh feature engineering: `phongtro_hanoi_all.csv` có **6.825** dòng; `phongtro_hanoi_features.csv` sau tái tạo có **6.825** dòng; `phongtro_hanoi_ml_ready.csv` còn **6.268** dòng sau làm sạch.
- Trong nhánh ML-ready, bước làm sạch loại **54** dòng có giá/diện tích thiếu hoặc không hợp lệ, **424** dòng ngoại lai diện tích và **79** dòng ngoại lai giá.
- Nhánh production: `phongtro_hanoi_advanced_nlp.csv` có **6.569** dòng; sau lọc domain và khử trùng theo khóa đã sửa còn **6.366** dòng.

## 4. Kết quả huấn luyện production

Giá được tính theo **triệu VND**. Dữ liệu được chia theo thời gian; 80% đầu dùng để train và 20% cuối dùng để test.

- **R²:** train 0,485; test 0,364.
- **MAE:** train 0,638; test 0,759 triệu VND.
- **RMSE:** train 0,959; test 1,009 triệu VND.
- **MAPE:** train 26,386%; test 25,291%.

Model mới đã được lưu tại `models/pipeline_production.joblib`. Biểu đồ permutation importance được tạo tại `plots/4_feature_importance.png`.

Các metric trên là kết quả hiện tại. Chưa có metric baseline được tính bằng cùng dữ liệu và cùng mốc chia thời gian, nên báo cáo không kết luận model đã cải thiện độ chính xác so với bản trước.

## 5. Kiểm chứng

- `test_pipeline_fixes.py`: **đạt**.
- `test_spatial_features.py`: **đạt**.
- Kiểm tra cú pháp toàn bộ file Python: **đạt**.
- Chạy `feature_importance.py`: **đạt**, biểu đồ được tạo.
- Chạy `pipeline_production.py`: **đạt**, model và các metric train/test được sinh.
- Kiểm tra trực tiếp trên dữ liệu thật: **6.366** dòng sau xử lý production và không còn bản trùng theo khóa mới.

## 6. Bản backup và sản phẩm

Bản backup được tạo trước khi sửa tại:

`C:\Users\NAME\Documents\demo_dap_backup_2026-10-03`

Các file chính đã cập nhật hoặc tạo lại:

- `feature_engineering.py`
- `pipeline_production.py`
- `advanced_pipeline_prep.py`
- `crawl_data.py`
- `geocoding_poi.py`
- `eda_and_cleaning.py`
- `feature_importance.py`
- `test_pipeline_fixes.py`
- `test_spatial_features.py`
- `requirements.txt`
- `data/phongtro_hanoi_features.csv`
- `data/phongtro_hanoi_ml_ready.csv`
- `models/pipeline_production.joblib`
- `plots/4_feature_importance.png`

## 7. Việc cần làm tiếp

1. Chốt cách đánh giá phù hợp với lúc triển khai: dự đoán cuốn chiếu (được dùng dữ liệu các tin đã quan sát trước đó) hay dự đoán nhiều tin tương lai cùng lúc. Feature lịch sử hiện tại phù hợp với kịch bản cuốn chiếu; kịch bản dự đoán đồng thời cần cách dựng feature khác để tránh dùng giá thực tế trong tập test.
2. Tính baseline và model mới trên cùng một mốc thời gian, cùng tập test để có kết luận so sánh công bằng.
3. Bổ sung log, retry và thống kê tỷ lệ geocode lỗi; chưa chạy request tới Nominatim trong đợt kiểm tra này.

