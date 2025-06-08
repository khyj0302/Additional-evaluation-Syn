import os
import numpy as np
import pandas as pd
import gower
import matplotlib.pyplot as plt
from tqdm import tqdm

# === 1️⃣ 구간 중앙값 변환 함수 ===
def range_to_midpoint(value):
    try:
        if isinstance(value, str) and '-' in value:
            low, high = map(float, value.split('-'))
            return (low + high) / 2
        else:
            return float(value)
    except:
        return np.nan

# === 2️⃣ 전처리 함수 (diabetes 전용) ===
def preprocess_diabetes_df(df):
    df = df.copy()
    for col in ['Age', 'Tumor_size', 'Inv_nodes']:
        if col in df.columns:
            df[col] = df[col].apply(range_to_midpoint)
    yes_no_cols = ['Node_caps', 'Irradiat']
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map({'yes': 1, 'no': 0})
    df = df.replace('?', np.nan)
    return df

# === 🌟 샘플링 수 설정 ===
REAL_SAMPLE_SIZE = 6000

# === 3️⃣ real 데이터 로드 및 샘플링 ===
real_data_path = '/home/khyj/0. phd/DC_original_final/diabetes_original.csv'
real_df_full = pd.read_csv(real_data_path)
real_df = preprocess_diabetes_df(real_df_full)
real_df_sampled = real_df.sample(n=min(REAL_SAMPLE_SIZE, len(real_df)), random_state=42).reset_index(drop=True)

# === 4️⃣ real–real 최소 거리 평균 계산 ===
print(f"Real–real distance 계산 중 (샘플 {REAL_SAMPLE_SIZE}개)...")
real_distances = gower.gower_matrix(real_df_sampled)
np.fill_diagonal(real_distances, np.inf)
real_min_distances = np.min(real_distances, axis=1)
real_mean_min_distance = np.mean(real_min_distances)
print(f"Real–real mean min distance (샘플 {REAL_SAMPLE_SIZE}개): {real_mean_min_distance:.4f}")

# === 5️⃣ synthetic 데이터 폴더 순회 ===
synthetic_folder = '/home/khyj/0. phd/DC_diabetes_final'

for file in tqdm(os.listdir(synthetic_folder)):
    if file.endswith('.csv'):
        syn_path = os.path.join(synthetic_folder, file)
        syn_df_full = pd.read_csv(syn_path)
        syn_df = preprocess_diabetes_df(syn_df_full).reset_index(drop=True)

        print(f"{file} - Gower distance 계산 중...")

        # synthetic–real 거리 계산 (real 샘플 사용)
        syn_real_distances = gower.gower_matrix(syn_df, real_df_sampled)
        syn_min_distances = np.min(syn_real_distances, axis=1)

        # 기준선 이하 synthetic 레코드 개수
        risky_count = np.sum(syn_min_distances < real_mean_min_distance)
        print(f"{file}: {risky_count} risky records (below threshold)")

        # === 시각화 (기준선 없음) ===
        plt.figure(figsize=(8, 5))
        plt.hist(syn_min_distances, bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Min Distance Distribution: {file}')
        plt.xlabel('Min Distance')
        plt.ylabel('Frequency')
        plt.tight_layout()
        output_png = f"min_distance_distribution_nobase_{file.replace('.csv', '')}.png"
        plt.savefig(output_png)
        plt.close()
        print(f"히스토그램 저장 완료: {output_png}")
