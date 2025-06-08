import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gower

# === 🔧 구간값 중앙값으로 변환하는 함수 ===
def range_to_midpoint(value):
    if isinstance(value, str) and '-' in value:
        parts = value.split('-')
        if all(p.isdigit() for p in parts):
            return (int(parts[0]) + int(parts[1])) / 2
    try:
        return float(value)
    except:
        return value  # 숫자 변환 실패하면 원래 값 유지

# === 🔧 전처리 함수 (숫자 vs 범주형 분리) ===
def preprocess_liver_dataset(df):
    df_processed = df.copy()
    numeric_cols = []
    categorical_cols = []

    for col in df_processed.columns:
        if col in ['tx1_name', 'i_h_tnm_stage']:
            df_processed[col] = df_processed[col].astype(str)
        else:
            df_processed[col] = df_processed[col].apply(range_to_midpoint)
            numeric_cols.append(col)

    df_processed[numeric_cols] = df_processed[numeric_cols].apply(pd.to_numeric, errors='coerce')
    return df_processed

# === 1️⃣ 데이터 로드 ===
real_data_path = '/home/khyj/0. phd/TTGAN_dataset_final/original_final/original_liver.csv'   
synthetic_data_path = '/home/khyj/0. phd/TTGAN_dataset_final/syn_liver/synthetic_liver_TTGAN_LGBM.csv'  

real_df = pd.read_csv(real_data_path)
synthetic_df = pd.read_csv(synthetic_data_path)

# === ✅ 전처리 적용 ===
real_df = preprocess_liver_dataset(real_df)
synthetic_df = preprocess_liver_dataset(synthetic_df)

# === 2️⃣ Gower distance 계산 ===
print("Gower distance 계산 중...")
distances = gower.gower_matrix(synthetic_df, real_df)
min_distances = np.min(distances, axis=1)

# === 3️⃣ DCR A score 계산 ===
print("real–real 기준선 계산 중...")
real_distances = gower.gower_matrix(real_df)
np.fill_diagonal(real_distances, np.inf)
mean_real_min_distance = np.mean(np.min(real_distances, axis=1))
A_score = np.mean(min_distances < mean_real_min_distance)
print(f"DCR A score: {A_score:.4f}")

# === 4️⃣ min distance 분포 시각화 (기준선 제거) ===
plt.figure(figsize=(8, 5))
plt.hist(min_distances, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Synthetic–Real Min Distances')
plt.xlabel('Min Distance')
plt.ylabel('Frequency')
plt.tight_layout()
output_path = 'min_distance_distribution_TTGAN_liver.png'
plt.savefig(output_path)
plt.show()

print(f"히스토그램 저장 완료: {output_path}")
