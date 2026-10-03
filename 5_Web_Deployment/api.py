import pandas as pd
import numpy as np
import joblib
import math
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from geopy.geocoders import Nominatim
from sklearn.neighbors import BallTree

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '5_Web_Deployment' in os.path.abspath(__file__) else os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

app = FastAPI()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/plots", StaticFiles(directory=PLOTS_DIR), name="plots")

geolocator = Nominatim(user_agent="hanoi_room_ai")

# Load assets
try:
    model_bundle = joblib.load(os.path.join(MODELS_DIR, 'pipeline_production.joblib'))
    df_main = pd.read_csv(os.path.join(DATA_DIR, 'phongtro_hanoi_advanced_nlp.csv'))
    df_main['price'] = df_main['price'] / 1_000_000.0  # Chuyển đổi sang đơn vị Triệu VNĐ
    df_geo = pd.read_csv(os.path.join(DATA_DIR, 'geocoded_addresses_checkpoint.csv'))[['exact_address', 'lat', 'lon']].dropna()
    df_geo = df_geo.drop_duplicates(subset=['exact_address'], keep='first')
    df = pd.merge(df_main, df_geo, on='exact_address', how='left')
    df = df.dropna(subset=['lat', 'lon']).reset_index(drop=True)
    # Rebuild tree on the fly
    tree = BallTree(np.radians(df[['lat', 'lon']].values), metric='haversine')
except Exception as e:
    print(f"Lỗi tải mô hình: {e}")

class RoomInput(BaseModel):
    address: str = ""
    district: str
    property_type: str
    area: float
    has_private_bathroom: bool
    has_private_entrance: bool
    is_furnished: bool
    living_with_owner: bool

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))

@app.post("/api/predict")
def predict_price(item: RoomInput):
    clean_district = item.district.replace(", Hà Nội", "").strip()
    full_address = f"{item.address}, {clean_district}, Hà Nội, Việt Nam"
    lat, lon, is_missing = np.nan, np.nan, 1.0
    
    try:
        district_center = geolocator.geocode(f"{clean_district}, Hà Nội, Việt Nam", timeout=5)
        location = geolocator.geocode(full_address, timeout=5)
        
        if not location and item.address.strip():
            loose_address = f"{item.address}, Hà Nội, Việt Nam"
            location = geolocator.geocode(loose_address, timeout=5)
            if location and len(location.address.split(',')) <= 3: 
                location = None

        if not location:
            location = district_center
            
        if not location:
            raise HTTPException(status_code=400, detail=f"Không tìm thấy khu vực '{clean_district}'.")
            
        lat, lon = location.latitude, location.longitude
        is_missing = 0.0
        
        # Validate Distance
        if district_center:
            dist_km = haversine(lat, lon, district_center.latitude, district_center.longitude)
            if dist_km > 15.0:
                raise HTTPException(status_code=400, detail=f"Xung đột địa lý: Địa chỉ này cách trung tâm '{clean_district}' {dist_km:.1f}km. Vui lòng chọn lại đúng Quận/Huyện!")
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Lỗi kết nối máy chủ bản đồ vệ tinh. Vui lòng thử lại sau!")
        
    # Dynamic District Fallback Median
    district_data = df[df['district'].str.contains(clean_district, na=False)]
    fallback_med_price = district_data['price'].median() if not district_data.empty else 4.0
    
    med_price, count = fallback_med_price, 0.0
    if not is_missing:
        dist, ind = tree.query(np.radians([[lat, lon]]), k=10)
        # haversine metric returns distance in radians. multiply by earth radius to get km -> km / 6371
        valid = dist[0] <= (0.5 / 6371.0) # 500m
        valid_idx = ind[0][valid]
        if len(valid_idx) > 0:
            med_price = df.iloc[valid_idx]['price'].median()
            count = len(valid_idx)

    try:
        # Build DataFrame
        input_data = pd.DataFrame({
            'district': [item.district],
            'property_type': [item.property_type],
            'posted_month': ['10'],
            'area': [item.area],
            'has_private_bathroom': [1.0 if item.has_private_bathroom else np.nan],
            'has_private_entrance': [1.0 if item.has_private_entrance else np.nan],
            'living_with_owner': [1.0 if item.living_with_owner else np.nan],
            'is_furnished': [1.0 if item.is_furnished else np.nan],
            'nearby_median_price_500m': [med_price],
            'nearby_listing_count_500m': [count],
            'is_geocode_missing': [is_missing],
            'time_idx': [3100]
        })
        
        pipeline = model_bundle['pipeline']
        trend_model = model_bundle['trend']
        features = model_bundle['features']
        
        X = input_data[features]
        X_time = input_data[['time_idx']]
        
        trend_pred = trend_model.predict(X_time)
        residual_pred = pipeline.predict(X)
        pred = trend_pred[0] + residual_pred[0]
        
        pred = max(0.5, float(pred))
        range_min = round(pred * 0.85, 2)
        range_max = round(pred * 1.15, 2)
        
        formatted_med = f"{med_price:.2f}".rstrip('0').rstrip('.').replace('.', ',')
        
        explanation = []
        explanation.append({"factor": f"Diện tích {item.area}m2", "impact": "++" if item.area > 30 else ("+" if item.area > 20 else "-")})
        explanation.append({"factor": f"Mặt bằng giá quanh khu vực", "impact": f"Giá tham chiếu: {formatted_med} Tr"})
        if item.has_private_bathroom: explanation.append({"factor": "Vệ sinh khép kín", "impact": "+"})
        if item.is_furnished: explanation.append({"factor": "Đầy đủ nội thất", "impact": "+"})
        
        return {
            "predicted_price": round(pred, 2),
            "range_min": range_min,
            "range_max": range_max,
            "geo_debug": {"lat": lat, "lon": lon, "neighbors": int(count), "med_price": formatted_med},
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
