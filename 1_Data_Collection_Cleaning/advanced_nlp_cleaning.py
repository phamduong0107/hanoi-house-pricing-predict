import pandas as pd
import numpy as np
import re

def extract_deep_features(row):
    t = str(row['description']).lower()
    
    # 1. Tiện ích Không gian & Quyền riêng tư (Sử dụng NaN nếu không nhắc đến)
    has_private_bathroom = 1.0 if any(k in t for k in ['khép kín', 'wc riêng', 'vệ sinh riêng', 'nhà vệ sinh riêng']) else np.nan
    has_private_entrance = 1.0 if any(k in t for k in ['lối đi riêng', 'chìa khóa riêng', 'giờ giấc tự do', 'không chung chủ']) else np.nan
    living_with_owner = 1.0 if 'chung chủ' in t and 'không chung chủ' not in t else np.nan
    is_furnished = 1.0 if any(k in t for k in ['full đồ', 'đủ đồ', 'nội thất đầy đủ', 'full nội thất', 'đầy đủ đồ']) else np.nan
    
    # 2. Chi phí ngầm (Giá điện, Giá nước, Phí dịch vụ)
    # Quy tắc: Điện (3k-5k), Nước (20k-150k), Dịch vụ (50k-300k). Lọc bỏ giá trị ảo.
    elec_match = re.search(r'điện\s*[:\s]*(\d+(?:[.,]\d+)?)\s*[kK]', t)
    elec = float(elec_match.group(1).replace(',', '.')) if elec_match else np.nan
    if elec > 10: elec = np.nan 
    
    water_match = re.search(r'nước\s*[:\s]*(\d+(?:[.,]\d+)?)\s*[kK]', t)
    water = float(water_match.group(1).replace(',', '.')) if water_match else np.nan
    if water > 200: water = np.nan 
    
    service_match = re.search(r'(?:dịch vụ|dv)\s*[:\s]*(\d+(?:[.,]\d+)?)\s*[kK]', t)
    service = float(service_match.group(1).replace(',', '.')) if service_match else np.nan
    if service > 500: service = np.nan
    
    # 3. Phân loại cấu trúc Bất động sản (Tránh False Positives)
    # Lược bỏ các cụm từ chỉ vị trí tương đối trước khi xét loại hình
    t_clean = re.sub(r'(?i)(gần|cách|kế bên|cạnh|sát|đối diện|khu vực|sau lưng|hướng đi)\s+(ccmn|chung cư mini|studio|căn hộ|nhà riêng|nguyên căn)', '', t)
    
    prop_type = 'phong_tro_thuong'
    if any(k in t_clean for k in ['ccmn', 'chung cư mini', 'studio', 'căn hộ']):
        prop_type = 'ccmn_studio'
    elif any(k in t_clean for k in ['nguyên căn', 'nhà riêng']):
        prop_type = 'nha_nguyen_can'
        
    return pd.Series([
        has_private_bathroom, has_private_entrance, living_with_owner, 
        is_furnished, elec, water, service, prop_type
    ])

def main():
    print("--- 1. LOADING DATA ---")
    df = pd.read_csv('data/phongtro_hanoi_all.csv')
    initial_len = len(df)

    print("--- 2. AUDIT: DEDUPLICATION (PREVENT REPOST LEAKAGE) ---")
    # Lọc bài đăng trùng lặp (cùng địa chỉ, diện tích, giá) để tránh việc Train và Test dính chung 1 nhà.
    df['posted_at'] = pd.to_datetime(df['posted_at'], errors='coerce')
    df = df.sort_values('posted_at', na_position='last', kind='mergesort')
    df = df.drop_duplicates(subset=['exact_address', 'area', 'price', 'district'])
    print(f" -> Removed {initial_len - len(df)} duplicate/repost listings.")

    print("--- 3. DEEP NLP EXTRACTION (NO TF-IDF RAW) ---")
    nlp_cols = [
        'has_private_bathroom', 'has_private_entrance', 'living_with_owner', 
        'is_furnished', 'electricity_price', 'water_price', 'service_fee', 'property_type'
    ]
    df[nlp_cols] = df.apply(extract_deep_features, axis=1)
    print(" -> Extracted hidden costs and property structures (CCMN/Studio).")

    print("--- 4. AUDIT: TEMPORAL SPLIT PREPARATION ---")
    # Sắp xếp theo ngày đăng để lát nữa thực hiện T-Split (Quá khứ -> Tương lai)
    df = df.sort_values('posted_at', na_position='last', kind='mergesort').reset_index(drop=True)
    print(" -> Data sorted by timeline for realistic Temporal Split.")

    print("--- 5. PRECISE GEOCODING PREPARATION ---")
    # Trích xuất địa chỉ UNIQUE để cache, giúp tiết kiệm 90% thời gian gọi API Geocoding
    unique_addresses = df[['exact_address', 'district']].dropna().drop_duplicates()
    unique_addresses.to_csv('data/unique_addresses_to_geocode.csv', index=False)
    print(f" -> Reduced {len(df)} rows to {len(unique_addresses)} UNIQUE addresses for API Geocoding.")

    df.to_csv('data/phongtro_hanoi_advanced_nlp.csv', index=False)
    print("\n[OK] Saved data/phongtro_hanoi_advanced_nlp.csv, ready for Ablation Study!")

if __name__ == '__main__':
    main()
