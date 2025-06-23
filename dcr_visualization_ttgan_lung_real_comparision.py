import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import gower
import os

# === 1️⃣ 데이터 로드 ===
real_data_path = '/home/khyj/0. phd/TTGAN_dataset_final/original_final/original_lung.csv' 
synthetic_data_path = '/home/khyj/0. phd/TTGAN_dataset_final/syn_lung/synthetic_lung_TTGAN_LGBM.csv' 

real_df = pd.read_csv(real_data_path)
synthetic_df = pd.read_csv(synthetic_data_path)

# === 2️⃣ Gower distance 계산 (Synthetic–Real) ===
print("Gower distance 계산 중 (Synthetic–Real)...")
distances = gower.gower_matrix(synthetic_df, real_df)
min_distances = np.min(distances, axis=1)

# === 3️⃣ Real–Real 거리 계산 및 DCR A Score ===
print("Gower distance 계산 중 (Real–Real)...")
real_distances = gower.gower_matrix(real_df)
np.fill_diagonal(real_distances, np.inf)
real_min_distances = np.min(real_distances, axis=1)
mean_real_min_distance = np.mean(real_min_distances)
A_score = np.mean(min_distances < mean_real_min_distance)
print(f"DCR A score: {A_score:.4f}")

# === 4️⃣ 히스토그램 (x축 + y축 고정) ===
plt.figure(figsize=(10, 6))

# ✅ x축 고정
x_min = 0.0
x_max = 0.22
bins = np.linspace(x_min, x_max, 31)

# ✅ y축 최대값 계산
real_counts, _ = np.histogram(real_min_distances, bins=bins)
syn_counts, _ = np.histogram(min_distances, bins=bins)
ymax = max(real_counts.max(), syn_counts.max()) * 1.1  # 10% buffer

# ✅ 히스토그램 그리기
plt.hist(real_min_distances, bins=bins, color='salmon', edgecolor='black', alpha=0.3, label='Real–Real')
plt.hist(min_distances, bins=bins, color='skyblue', edgecolor='black', alpha=0.5, label='Synthetic–Real')

plt.title('Min Distance Distribution: TTGAN')
plt.xlabel('Min Distance')
plt.ylabel('Frequency')
plt.ylim(0, ymax)  # ✅ y축 고정
plt.legend()
plt.tight_layout()

# 저장
output_path = 'min_distance_comparison_lung_TTGAN.png'
plt.savefig(output_path)
plt.close()
print(f"히스토그램 저장 완료: {output_path}")
