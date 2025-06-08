import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gower
import os
from tqdm import tqdm

# === 1️⃣ 데이터 로드 ===
real_data_path = '/home/khyj/0. phd/TTGAN_dataset_final/original_final/original_lung.csv' 
synthetic_data_path = '/home/khyj/0. phd/TTGAN_dataset_final/syn_lung/synthetic_lung_CopulaGAN.csv' 

real_df = pd.read_csv(real_data_path)
synthetic_df = pd.read_csv(synthetic_data_path)

# === 2️⃣ Gower distance 계산 ===
distances = gower.gower_matrix(synthetic_df, real_df)

# synthetic 각 행에서 real 데이터 중 최소 거리 추출
min_distances = np.min(distances, axis=1)

# === 3️⃣ DCR score 계산 ===
real_distances = gower.gower_matrix(real_df)
np.fill_diagonal(real_distances, np.inf)
mean_real_min_distance = np.mean(np.min(real_distances, axis=1))
A_score = np.mean(min_distances < mean_real_min_distance)
print(f"DCR A score: {A_score:.4f}")

# === 4️⃣ min distance 분포 시각화 (기준선 제거됨) ===
plt.figure(figsize=(8, 5))
plt.hist(min_distances, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Synthetic–Real Min Distances')
plt.xlabel('Min Distance')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('min_distance_distribution_lung_CopulaGAN.png')
plt.show()

print("히스토그램 저장 완료: min_distance_distribution_CopulaGAN.png")
