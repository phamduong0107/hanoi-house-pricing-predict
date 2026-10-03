import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
import os

def main():
    print("Loading data...")
    df = pd.read_csv("data/phongtro_hanoi_ml_ready.csv")
    if "posted_at" in df.columns:
        df["posted_at"] = pd.to_datetime(df["posted_at"], errors="coerce")
        df = df.sort_values("posted_at", na_position="last", kind="mergesort").reset_index(drop=True)

    drop_cols = ['price', 'log_price', 'post_id', 'title', 'price_raw', 'description', 'exact_address', 'posted_at', 'expires_at', 'updated_at', '_url', 'listing_status']
    y = df['price']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    categorical_features = ['district']
    numeric_features = ['area', 'month', 'listing_count_by_district', 'loo_mean_price_by_district', 'median_price_by_district', 'median_price_by_area_range']
    numeric_features = [c for c in numeric_features if c in X.columns]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    model = TransformedTargetRegressor(
        regressor=HistGradientBoostingRegressor(random_state=42), 
        func=np.log1p, inverse_func=np.expm1
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Calculating permutation importance (this might take a few seconds)...")
    # Avoid fragile worker-process startup on Windows and managed runtimes.
    result = permutation_importance(pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1)
    
    sorted_idx = result.importances_mean.argsort()
    
    # Lấy top 15 features quan trọng nhất
    top_n = 15
    sorted_idx_top = sorted_idx[-top_n:]

    plt.figure(figsize=(12, 8))
    # Chuyển tên cột thành dạng danh sách để vẽ
    feature_names = np.array(X.columns)[sorted_idx_top]
    
    sns.boxplot(data=result.importances[sorted_idx_top].T, orient="h")
    plt.yticks(range(len(feature_names)), feature_names)
    plt.title("Feature Importance (Permutation) - HistGradientBoosting")
    plt.xlabel("Mức độ ảnh hưởng (Làm giảm sai số)")
    
    os.makedirs("plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig("plots/4_feature_importance.png")
    print("Done! Saved plot to plots/4_feature_importance.png")

if __name__ == "__main__":
    main()

