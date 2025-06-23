import os
import numpy as np
import pandas as pd
import gower
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 

# === 1️⃣ 기준 데이터 로드 ===
real_data_path = '/home/khyj/0. phd/DC_original_final/breast_original.csv'
real_df = pd.read_csv(real_data_path)

# === 2️⃣ real–real 최소 거리 계산 ===
print("Real–Real Gower distance 계산 중...")
real_distances = gower.gower_matrix(real_df)
np.fill_diagonal(real_distances, np.inf)
real_min_distances = np.min(real_distances, axis=1)
real_mean_min_distance = np.mean(real_min_distances)
print(f"Real–Real mean min distance: {real_mean_min_distance:.4f}")

# === 3️⃣ synthetic 데이터 (단일 파일) 설정 ===
synthetic_data_path = '/home/khyj/0. phd/DC_breast_final/dc_CTGAN_e100.csv'
syn_df = pd.read_csv(synthetic_data_path)

# === 4️⃣ synthetic–real 거리 계산 ===
print("Synthetic–Real Gower distance 계산 중...")
syn_real_distances = gower.gower_matrix(syn_df, real_df)
syn_min_distances = np.min(syn_real_distances, axis=1)

# === 5️⃣ A-score 및 risky 개수 출력 ===
risky_count = np.sum(syn_min_distances < real_mean_min_distance)
A_score = np.mean(syn_min_distances < real_mean_min_distance)
print(f"DCR A score: {A_score:.4f} ({risky_count} risky records)")

# === 6️⃣ 시각화 (x축 + y축 스케일 고정) ===
plt.figure(figsize=(10, 6))

# ✅ 고정된 x축 범위 및 bin 설정
x_min = 0.0
x_max = 0.22
bins = np.linspace(x_min, x_max, 31)  # 30개 구간

# ✅ y축 최대값 계산을 위한 빈도 추출
real_counts, _ = np.histogram(real_min_distances, bins=bins)
syn_counts, _ = np.histogram(syn_min_distances, bins=bins)
ymax = max(real_counts.max(), syn_counts.max()) * 1.1  # 10% 여유

# ✅ 히스토그램 그리기
plt.hist(real_min_distances, bins=bins, color='salmon', edgecolor='black', alpha=0.3, label='Real–Real')
plt.hist(syn_min_distances, bins=bins, color='skyblue', edgecolor='black', alpha=0.5, label='Synthetic–Real')

plt.title('Min Distance Distribution: MG CTGAN')
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
