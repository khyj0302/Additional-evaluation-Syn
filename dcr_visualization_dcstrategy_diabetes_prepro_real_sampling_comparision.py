import os
import numpy as np
import pandas as pd
import gower
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 백엔드 설정 (GUI 없는 환경에서 저장용)

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

# === 설정 ===
REAL_SAMPLE_SIZE = 6000
real_data_path = '/home/khyj/0. phd/DC_original_final/diabetes_original.csv'
synthetic_data_path = '/home/khyj/0. phd/DC_diabetes_final/dc_CTGAN_e100.csv'  # ⭐단일 파일만 처리

# === 3️⃣ real 데이터 로드 및 전처리 ===
real_df_full = pd.read_csv(real_data_path)
real_df = preprocess_diabetes_df(real_df_full)
real_df_sampled = real_df.sample(n=min(REAL_SAMPLE_SIZE, len(real_df)), random_state=42).reset_index(drop=True)

# === 4️⃣ real–real 거리 계산 ===
print(f"Real–real distance 계산 중 (샘플 {REAL_SAMPLE_SIZE}개)...")
real_distances = gower.gower_matrix(real_df_sampled)
np.fill_diagonal(real_distances, np.inf)
real_min_distances = np.min(real_distances, axis=1)
real_mean_min_distance = np.mean(real_min_distances)
print(f"Real–real mean min distance: {real_mean_min_distance:.4f}")

# === 5️⃣ synthetic 데이터 처리 ===
print(f"{synthetic_data_path} - Gower distance 계산 중...")
syn_df_full = pd.read_csv(synthetic_data_path)
syn_df = preprocess_diabetes_df(syn_df_full).reset_index(drop=True)

syn_real_distances = gower.gower_matrix(syn_df, real_df_sampled)
syn_min_distances = np.min(syn_real_distances, axis=1)

# DCR A score 계산
A_score = np.mean(syn_min_distances < real_mean_min_distance)
risky_count = np.sum(syn_min_distances < real_mean_min_distance)
print(f"DCR A score: {A_score:.4f} ({risky_count} risky records)")

# === 6️⃣ 시각화 (x축, y축 고정) ===
plt.figure(figsize=(10, 6))

# ✅ x축 범위 및 bin 설정
x_min = 0.0
x_max = 0.22
bins = np.linspace(x_min, x_max, 31)  # 30개 구간

# ✅ y축 범위 계산
real_counts, _ = np.histogram(real_min_distances, bins=bins)
syn_counts, _ = np.histogram(syn_min_distances, bins=bins)
ymax = max(real_counts.max(), syn_counts.max()) * 1.1  # 10% buffer

# ✅ 히스토그램 그리기
plt.hist(real_min_distances, bins=bins, color='salmon', edgecolor='black', alpha=0.3, label='Real–Real')
plt.hist(syn_min_distances, bins=bins, color='skyblue', edgecolor='black', alpha=0.5, label='Synthetic–Real')

plt.title('Min Distance Distribution: DC CTGAN')
plt.xlabel('Min Distance')
plt.ylabel('Frequency')
plt.ylim(0, ymax)  # ✅ y축 고정
plt.legend()
plt.tight_layout()

# === 7️⃣ 이미지 저장 ===
output_png = f"min_distance_comparison_{os.path.basename(synthetic_data_path).replace('.csv', '')}.png"
plt.savefig(output_png)
plt.close()
print(f"히스토그램 저장 완료: {output_png}")
