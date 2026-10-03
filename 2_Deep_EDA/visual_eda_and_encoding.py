import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Tạo thư mục lưu biểu đồ
os.makedirs("plots", exist_ok=True)

print("Loading ML-ready data...")
df = pd.read_csv("data/phongtro_hanoi_ml_ready.csv")

# ==========================================
# 1. VISUAL EDA (VẼ BIỂU ĐỒ KHẢO SÁT)
# ==========================================
print("Plotting Price Distribution...")
plt.figure(figsize=(10, 6))
sns.histplot(df["price"], bins=50, kde=True, color="blue")
plt.title("Phân phối Giá thuê phòng trọ (VND) - Đã cắt Outlier")
plt.xlabel("Giá thuê")
plt.ylabel("Số lượng")
plt.ticklabel_format(style='plain', axis='x')
plt.savefig("plots/1_price_distribution.png", bbox_inches='tight')
plt.close()

print("Plotting Boxplot by District...")
plt.figure(figsize=(14, 8))
# Chọn top 10 quận có nhiều bài đăng nhất để vẽ cho đẹp
top_districts = df['district'].value_counts().nlargest(10).index
sns.boxplot(x="district", y="price", data=df[df['district'].isin(top_districts)])
plt.title("Phân bổ giá thuê theo Top 10 Quận")
plt.xticks(rotation=45)
plt.ticklabel_format(style='plain', axis='y')
plt.savefig("plots/2_price_by_district_boxplot.png", bbox_inches='tight')
plt.close()

print("Plotting Correlation Heatmap...")
plt.figure(figsize=(16, 12))
# Chỉ chọn các biến số học và biến tiện ích
num_cols = df.select_dtypes(include=[np.number]).columns
corr = df[num_cols].corr()
# Chỉ hiển thị tương quan cao với log_price
target_corr = corr[['log_price']].sort_values(by='log_price', ascending=False)
sns.heatmap(target_corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mức độ ảnh hưởng của các Tiện ích tới Giá phòng (Correlation)")
plt.savefig("plots/3_feature_correlation.png", bbox_inches='tight')
plt.close()

# ==========================================
# 2. XỬ LÝ BIẾN PHÂN LOẠI (CATEGORICAL ENCODING)
# ==========================================
print("\nStarting Categorical Encoding...")

print(" - Applying One-Hot Encoding on 'district'...")
df = pd.get_dummies(df, columns=["district"], prefix="dist", drop_first=True)

# Các cột phân loại khác không cần dùng cho mô hình sẽ bị loại bỏ
cols_to_drop = ["post_id", "title", "price_raw", "description", "exact_address", "posted_at", "expires_at", "updated_at", "_url", "listing_status"]
existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
df.drop(columns=existing_cols_to_drop, inplace=True)

# Convert toàn bộ boolean (True/False sinh ra từ get_dummies) về 1/0
for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)

# ==========================================
# 3. LƯU BỘ DATA CUỐI CÙNG (FINAL FOR TRAINING)
# ==========================================
final_output = "data/phongtro_hanoi_final_train.csv"
df.to_csv(final_output, index=False, encoding="utf-8-sig")
print(f"\nDone! Final ML dataset saved to: {final_output}")
print("Check the 'plots' folder for Visual EDA images.")
