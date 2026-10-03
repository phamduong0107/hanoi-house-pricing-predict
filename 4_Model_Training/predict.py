import joblib
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def load_model(model_path='models/pipeline_production.joblib'):
    """Tải mô hình đã được huấn luyện từ ổ cứng."""
    return joblib.load(model_path)

def predict_price(model_bundle, input_data: pd.DataFrame) -> np.ndarray:
    """
    Dự đoán giá phòng trọ dựa trên dữ liệu đầu vào.
    Yêu cầu input_data phải có đủ các cột như lúc train (12 features).
    """
    pipeline = model_bundle['pipeline']
    trend_model = model_bundle['trend']
    features = model_bundle['features']
    
    # 1. Trích xuất đúng thứ tự cột
    X = input_data[features]
    X_time = input_data[['time_idx']]
    
    # 2. Dự đoán (Quy tắc Hybrid: Xu hướng + Phần dư)
    trend_pred = trend_model.predict(X_time)
    residual_pred = pipeline.predict(X)
    
    final_price_millions = trend_pred + residual_pred
    return final_price_millions

if __name__ == '__main__':
    print("Dang tai mo hinh AI...")
    model = load_model()
    print("Tai mo hinh thanh cong!\n")
    
    # Giả lập 1 khách hàng muốn định giá phòng trọ của họ
    print("Thong so can phong can dinh gia:")
    print("- Vi tri: Quan Cau Giay (Mat bang gia hang xom 500m: 4.5 Trieu)")
    print("- Loai hinh: Phong tro (25m2)")
    print("- Tien ich: Khep kin, Cua rieng, Day du noi that, Khong chung chu")
    print("- Thoi diem dang: Thang 8 (Mua tuu truong cao diem)")
    
    sample_listing = pd.DataFrame({
        'district': ['Quận Cầu Giấy'],
        'property_type': ['Phòng trọ'],
        'posted_month': ['8'], 
        'area': [25.0],
        'has_private_bathroom': [1.0],
        'has_private_entrance': [1.0],
        'living_with_owner': [np.nan], # Dùng NaN cho các thông tin không chắc chắn hoặc không chung chủ
        'is_furnished': [1.0],
        'nearby_median_price_500m': [4.5], 
        'nearby_listing_count_500m': [15.0],
        'is_geocode_missing': [0.0],
        'time_idx': [3100] # Tương đương mốc thời gian năm 2024-2025
    })
    
    predicted_prices = predict_price(model, sample_listing)
    
    print("-" * 50)
    print(f"MUC GIA DU DOAN (DE XUAT): {predicted_prices[0]:.2f} Trieu VND / Thang")
    print("-" * 50)
