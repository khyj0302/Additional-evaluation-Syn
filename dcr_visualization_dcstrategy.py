import os
import numpy as np
import pandas as pd
import gower
import matplotlib.pyplot as plt
from tqdm import tqdm

# 기준 데이터 로드
real_data_path = '/home/khyj/0. phd/DC_original_final/breast_original.csv'
real_df = pd.read_csv(real_data_path)

# real–real 최소 거리 평균 계산
real_distances = gower.gower_matrix(real_df)
np.fill_diagonal(real_distances, np.inf)
real_min_distances = np.min(real_distances, axis=1)
real_mean_min_distance = np.mean(real_min_distances)
print(f"Real–real mean min distance: {real_mean_min_distance:.4f}")

# synthetic 데이터 폴더
synthetic_folder = '/home/khyj/0. phd/DC_breast_final'

# 순회
for file in tqdm(os.listdir(synthetic_folder)):
    if file.endswith('.csv'):
        syn_path = os.path.join(synthetic_folder, file)
        syn_df = pd.read_csv(syn_path)

        # synthetic–real 거리 계산
        syn_real_distances = gower.gower_matrix(syn_df, real_df)
        syn_min_distances = np.min(syn_real_distances, axis=1)

        # 기준선 이하 synthetic 레코드 개수 (출력은 유지)
        risky_count = np.sum(syn_min_distances < real_mean_min_distance)
        print(f"{file}: {risky_count} risky records (below threshold)")

        # === 시각화 === (기준선 제거됨)
        plt.figure(figsize=(8, 5))
        plt.hist(syn_min_distances, bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Synthetic–Real Min Distance: {file}')
        plt.xlabel('Min Distance')
        plt.ylabel('Frequency')
        plt.tight_layout()
        output_png = f"min_distance_distribution_nobase_{file.replace('.csv', '')}.png"
        plt.savefig(output_png)
        plt.close()
        print(f"히스토그램 저장 완료: {output_png}")
