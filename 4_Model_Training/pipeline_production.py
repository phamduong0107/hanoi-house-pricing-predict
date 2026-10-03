import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics.pairwise import haversine_distances
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
import joblib
import warnings
import logging
import argparse
from typing import Tuple, List

warnings.filterwarnings('ignore')

# Cấu hình Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_and_clean_data(data_path: str, geo_path: str) -> pd.DataFrame:
    """Tải và làm sạch dữ liệu với bộ lọc kiến thức miền."""
    logger.info(f"Đang tải dữ liệu từ {data_path} và {geo_path}...")
    df = pd.read_csv(data_path)
    df_geo = pd.read_csv(geo_path)[['exact_address', 'lat', 'lon']].dropna(subset=['lat', 'lon'])
    df_geo = df_geo.drop_duplicates(subset=['exact_address'], keep='first')
    
    df = pd.merge(df, df_geo, on='exact_address', how='left')
    
    # Domain-based Filtering & Unit Scaling
    initial_len = len(df)
    df = df[(df['price'] >= 500000) & (df['price'] <= 25000000) & (df['area'] >= 10) & (df['area'] <= 100)]
    df['price'] = df['price'] / 1_000_000.0 
    logger.info(f"Lọc Domain Knowledge: Xóa {initial_len - len(df)} bản ghi ngoại lai.")
    
    # Thời gian & Deduplication
    df['posted_at'] = pd.to_datetime(df['posted_at'], errors='coerce')
    df['posted_date'] = df['posted_at'].dt.date
    
    len_before_dedup = len(df)
    df = df.drop_duplicates(subset=['exact_address', 'price', 'posted_date'], keep='first')
    logger.info(f"Deduplication: Đã xóa {len_before_dedup - len(df)} bản ghi trùng lặp (spam).")
    
    df = df.sort_values('posted_at', na_position='last', kind='mergesort').reset_index(drop=True)
    
    # Kỹ thuật Đặc trưng cơ bản
    df['time_idx'] = (df['posted_at'] - df['posted_at'].min()).dt.days.fillna(0)
    df['posted_month'] = df['posted_at'].dt.month.astype(str)
    df['is_geocode_missing'] = df['lat'].isna().astype(float)
    
    # Xử lý biến NLP (Không đề cập != Không có)
    nlp_cols = ['has_private_bathroom', 'has_private_entrance', 'living_with_owner', 'is_furnished']
    df[nlp_cols] = df[nlp_cols].replace(0.0, np.nan)
    
    return df

def add_time_aware_spatial_features(df: pd.DataFrame, radius_km: float = 0.5, time_window_days: int = 180) -> pd.DataFrame:
    """Xây dựng đặc trưng không gian, kiểm soát rò rỉ tương lai với thuật toán BallTree tối ưu O(N log N)."""
    logger.info("Đang tính toán đặc trưng Không gian-Thời gian (BallTree 500m)...")
    valid_geo = df['lat'].notna().values
    df_latlon = df[['lat', 'lon']].fillna(df[['lat', 'lon']].median())
    latlon_rad = np.radians(df_latlon.values)
    
    # Tối ưu hóa: Dùng BallTree thay vì tính ma trận khoảng cách N x N
    from sklearn.neighbors import BallTree
    tree = BallTree(latlon_rad, metric='haversine')
    r_rad = radius_km / 6371.0
    neighbors_list = tree.query_radius(latlon_rad, r=r_rad)
    
    prices = df['price'].values
    posted_times = df['posted_at'].values
    
    N = len(df)
    med_price = np.full(N, np.nan)
    count = np.full(N, np.nan)
    
    for i in range(N):
        if not valid_geo[i]:
            continue
            
        # Lấy danh sách index các hàng xóm trong bán kính
        idx = neighbors_list[i]
        
        # Tính khoảng cách thời gian (Ngày)
        time_diffs = (posted_times[i] - posted_times[idx]).astype('timedelta64[D]').astype(float)
        
        # Lọc hàng xóm: Phải đăng trước đó, không quá 180 ngày, và có tọa độ hợp lệ
        valid_mask = (time_diffs > 0) & (time_diffs <= time_window_days) & valid_geo[idx]
        valid_idx = idx[valid_mask]
        
        count[i] = len(valid_idx)
        if count[i] > 0:
            med_price[i] = np.median(prices[valid_idx])

    df['nearby_median_price_500m'] = med_price
    df['nearby_listing_count_500m'] = count
    return df

def detrend_and_train_model(df: pd.DataFrame) -> Tuple[Pipeline, LinearRegression, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Tách xu hướng tuyến tính và huấn luyện HistGBM trên phần dư."""
    logger.info("Đang huấn luyện mô hình Hybrid Detrending...")
    N = len(df)
    split_idx = int(N * 0.8)
    
    cat_features = ['district', 'property_type', 'posted_month']
    num_features = ['area', 'has_private_bathroom', 'has_private_entrance', 'living_with_owner', 'is_furnished', 'nearby_median_price_500m', 'nearby_listing_count_500m', 'is_geocode_missing']
    all_features = cat_features + num_features
    
    X_time_train, X_time_test = df[['time_idx']].iloc[:split_idx], df[['time_idx']].iloc[split_idx:]
    X_train, X_test = df[all_features].iloc[:split_idx], df[all_features].iloc[split_idx:]
    y_train, y_test = df['price'].iloc[:split_idx].values, df['price'].iloc[split_idx:].values
    
    # Linear Detrending
    trend_model = LinearRegression().fit(X_time_train, y_train)
    residual_train = y_train - trend_model.predict(X_time_train)
    
    # HistGBM
    preprocessor = ColumnTransformer(
        [('ord_enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_features)],
        remainder='passthrough'
    )
    
    histgbm = HistGradientBoostingRegressor(
        categorical_features=[0, 1, 2],
        learning_rate=0.05, max_iter=100, max_leaf_nodes=15, 
        min_samples_leaf=20, l2_regularization=1.0, random_state=42
    )
    
    pipeline = Pipeline([('prep', preprocessor), ('model', histgbm)])
    pipeline.fit(X_train, residual_train)
    
    pred_train = trend_model.predict(X_time_train) + pipeline.predict(X_train)
    pred_test = trend_model.predict(X_time_test) + pipeline.predict(X_test)
    
    return pipeline, trend_model, y_train, pred_train, y_test, pred_test, all_features

def evaluate_and_save(pipeline: Pipeline, trend_model: LinearRegression, features: List[str], y_train: np.ndarray, pred_train: np.ndarray, y_test: np.ndarray, pred_test: np.ndarray, output_path: str):
    """Đánh giá mô hình và lưu ra đĩa."""
    logger.info("Đang đánh giá mô hình...")
    metrics = {
        'R2': (r2_score(y_train, pred_train), r2_score(y_test, pred_test)),
        'MAE (Trieu)': (mean_absolute_error(y_train, pred_train), mean_absolute_error(y_test, pred_test)),
        'RMSE (Trieu)': (np.sqrt(mean_squared_error(y_train, pred_train)), np.sqrt(mean_squared_error(y_test, pred_test))),
        'MAPE (%)': (mean_absolute_percentage_error(y_train, pred_train)*100, mean_absolute_percentage_error(y_test, pred_test)*100)
    }
    
    print(f"\n{'Metric':<15} | {'Train Set':<15} | {'Test Set':<15}")
    print("-" * 50)
    for name, (tr, te) in metrics.items():
        print(f"{name:<15} | {tr:.3f}{'%' if '%' in name else '':<14} | {te:.3f}{'%' if '%' in name else ''}")
        
    joblib.dump({'pipeline': pipeline, 'trend': trend_model, 'features': features}, output_path)
    logger.info(f"[SUCCESS] Mô hình đã được đóng gói tại {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Định giá BĐS Hà Nội.")
    parser.add_argument('--data', type=str, default='data/phongtro_hanoi_advanced_nlp.csv', help='Đường dẫn file dữ liệu.')
    parser.add_argument('--geo', type=str, default='data/geocoded_addresses_checkpoint.csv', help='Đường dẫn file tọa độ.')
    parser.add_argument('--output', type=str, default='models/pipeline_production.joblib', help='Đường dẫn xuất mô hình.')
    args = parser.parse_args()
    
    df_clean = load_and_clean_data(args.data, args.geo)
    df_spatial = add_time_aware_spatial_features(df_clean)
    pipe, trend, y_tr, p_tr, y_te, p_te, feats = detrend_and_train_model(df_spatial)
    evaluate_and_save(pipe, trend, feats, y_tr, p_tr, y_te, p_te, args.output)
