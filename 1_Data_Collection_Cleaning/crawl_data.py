import os
import re
import time
import argparse
import requests
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://phongtro123.com"
START_URL = f"{BASE_URL}/tinh-thanh/ha-noi"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
MAX_WORKERS = 5
MAX_RETRIES = 3

FEATURES = {
    "has_water_heater": ["nóng lạnh", "máy nóng lạnh"],
    "has_washing_machine": ["máy giặt"],
    "has_kitchen": ["bếp", "nấu bếp"],
    "has_bed": ["giường"],
    "has_air_conditioner": ["điều hòa", "máy lạnh"],
    "has_parking": ["chỗ để xe", "để xe", "đỗ xe"],
    "has_refrigerator": ["tủ lạnh"],
    "has_wardrobe": ["tủ quần áo"],
    "has_balcony": ["ban công"],
    "has_loft": ["gác", "gác xép"],
    "has_elevator": ["thang máy"],
    "has_drying_area": ["phơi đồ", "sân phơi"],
    "has_desk": ["bàn học", "bàn làm việc"],
    "is_ccmn": ["ccmn", "chung cư mini", "căn hộ dịch vụ", "studio"],
    "is_chung_chu": ["chung chủ", "cùng chủ", "sống cùng chủ"]
}

def get_soup(url):
    for i in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            if i == MAX_RETRIES - 1:
                print(f"Failed to fetch {url}: {e}")
                return None
            time.sleep(1)
    return None

def get_total_pages(soup):
    if not soup:
        return 1
    pages = [
        int(m.group(1))
        for a in soup.find_all("a", href=True)
        if (m := re.search(r"[?&]page=(\d+)", a["href"]))
    ]
    return max(pages, default=1)

def crawl_listings(max_pages=None):
    print("Fetching total pages...")
    soup = get_soup(START_URL)
    last_page = get_total_pages(soup)
    
    if max_pages and max_pages < last_page:
        last_page = max_pages

    print(f"Total pages to crawl: {last_page}")
    
    data = []
    seen = set()

    for page in range(1, last_page + 1):
        url = START_URL if page == 1 else f"{START_URL}?page={page}"
        soup = get_soup(url)
        if not soup:
            continue
            
        listings = soup.select("ul.post__listing > li")
        for li in listings:
            a = li.select_one("h3 a")
            if not a:
                continue

            href = a.get("href")
            if not href:
                continue

            detail_url = BASE_URL + href if href.startswith("/") else href
            
            btn = li.select_one("button[data-post-id]")
            post_id = btn.get("data-post-id") if btn else None
            key = post_id or detail_url

            if key in seen:
                continue
            seen.add(key)

            text = li.get_text(" ", strip=True)
            area_match = re.search(r"(\d+(?:[.,]\d+)?)\s*m(?:²|2)?", text)
            price_tag = li.select_one(".text-green")
            district_tag = li.select_one("a.text-body")

            data.append({
                "post_id": post_id,
                "title": a.get_text(" ", strip=True),
                "price_raw": price_tag.get_text(" ", strip=True) if price_tag else None,
                "district": district_tag.get_text(" ", strip=True) if district_tag else None,
                "area": area_match.group(1).replace(",", ".") if area_match else None,
                "_url": detail_url
            })
        print(f"Page {page}/{last_page} | Total items: {len(data)}")

    return pd.DataFrame(data)

def crawl_details_for_row(url):
    empty = {
        "description": None,
        "exact_address": None,
        "posted_at": None,
        "expires_at": None,
        "updated_at": None
    }
    soup = get_soup(url)
    if not soup:
        return empty

    text = soup.get_text(" ", strip=True)
    
    h2 = next((x for x in soup.find_all("h2") if "Thông tin mô tả" in x.get_text(" ", strip=True)), None)
    description = ""
    if h2:
        parts = []
        for tag in h2.find_all_next(["p", "h2"]):
            if tag.name == "h2":
                break
            t = tag.get_text(" ", strip=True)
            if t:
                parts.append(t)
        description = " ".join(parts)

    def find_date(label):
        m = re.search(rf"{label}.*?(\d{{1,2}}:\d{{2}})\s*,?\s*(\d{{1,2}}/\d{{1,2}}/\d{{4}})", text, re.I)
        return f"{m.group(1)} {m.group(2)}" if m else None
        
    exact_address = None
    try:
        import json
        for script_tag in soup.find_all("script", type="application/ld+json"):
            if script_tag and script_tag.string:
                data = json.loads(script_tag.string)
                if "address" in data and "streetAddress" in data["address"]:
                    exact_address = data["address"]["streetAddress"]
                    break
    except Exception:
        pass

    return {
        "description": description,
        "exact_address": exact_address,
        "posted_at": find_date("Ngày đăng"),
        "expires_at": find_date("Ngày hết hạn"),
        "updated_at": find_date("Cập nhật")
    }

def enrich_with_details(df):
    if df.empty:
        return df

    details = [None] * len(df)
    urls = df["_url"].tolist()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        jobs = {pool.submit(crawl_details_for_row, url): i for i, url in enumerate(urls)}
        for n, future in enumerate(as_completed(jobs), 1):
            details[jobs[future]] = future.result()
            if n % 50 == 0 or n == len(df):
                print(f"Crawled details: {n}/{len(df)}")
                
    details_df = pd.DataFrame(details)
    df = pd.concat([df.drop(columns=["_url"]), details_df], axis=1)
    return df

def clean_price(x):
    if pd.isna(x):
        return None
    x = str(x).lower().replace(",", ".")
    if "triệu" in x:
        m = re.search(r"\d+(?:\.\d+)?", x)
        return float(m.group()) * 1_000_000 if m else None
    m = re.search(r"\d[\d.,]*", x)
    return float(re.sub(r"[.,]", "", m.group())) if m else None

def extract_feature(text, words):
    if pd.isna(text):
        return pd.NA
    text = str(text).lower()
    result = []
    for word in words:
        for m in re.finditer(re.escape(word), text):
            before = text[max(0, m.start() - 25):m.start()]
            # "không chung chủ", "không giường", "không có điều hòa", "chưa có tủ lạnh"
            negative = re.search(r"(không(\s*(có|dùng|sử dụng))?|chưa\s*có)\s*$", before)
            result.append(0 if negative else 1)
    if not result:
        return pd.NA
    if 0 in result and 1 in result:
        return pd.NA
    return max(result)

def process_and_clean_data(df):
    print("Cleaning and processing data...")
    # Clean price and area
    df["price"] = df["price_raw"].apply(clean_price)
    df["area"] = pd.to_numeric(df["area"], errors="coerce")

    # Clean dates
    for col in ["posted_at", "expires_at", "updated_at"]:
        df[col] = pd.to_datetime(df[col], format="%H:%M %d/%m/%Y", errors="coerce")

    # Status
    now = pd.Timestamp.now()
    df["listing_status"] = df["expires_at"].apply(
        lambda x: "active" if pd.notna(x) and x >= now else "expired"
    )
    df["description_length"] = df["description"].fillna("").str.len()

    # Features extraction
    for name, words in FEATURES.items():
        df[name] = df["description"].apply(lambda x: extract_feature(x, words)).astype("Int64")

    # Nhóm 4: Thời gian & Kinh tế
    print("Extracting Group 4 Features (Time & Economy)...")
    
    # 1. Tháng và Năm đăng bài
    df["month"] = df["posted_at"].dt.month.astype("Int64")
    df["year"] = df["posted_at"].dt.year.astype("Int64")
    
    # 2. Mùa vụ (Tháng 7, 8, 9 sinh viên nhập học)
    df["is_peak_season"] = df["month"].isin([7, 8, 9]).astype(int)
        
    return df

def main():
    parser = argparse.ArgumentParser(description="Crawl data phongtro123")
    parser.add_argument("--pages", type=int, default=None, help="Max pages to crawl (for testing)")
    args = parser.parse_args()

    df = crawl_listings(max_pages=args.pages)
    if df.empty:
        print("No data found!")
        return

    df = enrich_with_details(df)
    df = process_and_clean_data(df)

    os.makedirs("data", exist_ok=True)
    
    # Save all
    df.to_csv("data/phongtro_hanoi_all.csv", index=False, encoding="utf-8-sig")

    # Save current active
    current = df[
        (df["listing_status"] == "active") &
        (df["price"] > 0) &
        (df["area"] > 0) &
        ~df["district"].str.contains("Hồ Chí Minh", case=False, na=False)
    ].copy()
    current.to_csv("data/phongtro_hanoi_current.csv", index=False, encoding="utf-8-sig")

    print("\n========== DONE ==========")
    print(f"Total entries: {df.shape[0]}")
    print(f"Current active entries: {current.shape[0]}")
    print("\nStatus:")
    print(df["listing_status"].value_counts(dropna=False))
    print("\nMissing Dates:")
    print(df[["posted_at", "expires_at", "updated_at"]].isna().sum())
    print(f"\nDuplicate post_ids: {df['post_id'].duplicated().sum()}")

if __name__ == "__main__":
    main()