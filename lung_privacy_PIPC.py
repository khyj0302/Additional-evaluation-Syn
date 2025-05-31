import pandas as pd
import numpy as np
import gower
import os
from tqdm import tqdm

# === 데이터 전처리 (모든 컬럼 숫자화) ===
def preprocess_dataset(df):
    df_processed = df.copy()
    for col in df_processed.columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    return df_processed

# === 1️⃣ real–real 최소 거리 계산 ===
def compute_real_min_distances(real_df):
    real_matrix = gower.gower_matrix(real_df)
    np.fill_diagonal(real_matrix, np.inf)  # 자기 자신은 무시
    min_distances = np.min(real_matrix, axis=1)
    return min_distances

# === 2️⃣ synthetic–real 최소 거리 계산 ===
def compute_synthetic_min_distances(synthetic_df, real_df):
    syn_real_matrix = gower.gower_matrix(synthetic_df, real_df)
    min_distances = np.min(syn_real_matrix, axis=1)
    return min_distances

# === 3️⃣ 추론 위험도 A 계산 ===
def compute_inference_risk_A(real_df, synthetic_df):
    real_min_dists = compute_real_min_distances(real_df)
    synthetic_min_dists = compute_synthetic_min_distances(synthetic_df, real_df)
    count = np.sum(synthetic_min_dists < real_min_dists)
    A_score = count / len(synthetic_df)
    return A_score

# === 4️⃣ 전체 폴더 순회 ===
def evaluate_synthetic_folder(real_path, synthetic_folder):
    real_df = pd.read_csv(real_path)
    real_df = preprocess_dataset(real_df)
    results = []

    for file in tqdm(os.listdir(synthetic_folder)):
        if file.endswith('.csv'):
            synthetic_path = os.path.join(synthetic_folder, file)
            synthetic_df = pd.read_csv(synthetic_path)
            synthetic_df = preprocess_dataset(synthetic_df)

            try:
                inference_risk_A = compute_inference_risk_A(real_df, synthetic_df)
            except Exception as e:
                print(f"Error in {file}: {e}")
                inference_risk_A = np.nan

            results.append({
                'synthetic_file': file,
                'inference_risk_A': inference_risk_A
            })

    results_df = pd.DataFrame(results)
    return results_df

# === 5️⃣ 실행 예시 ===
real_data_path = '/home/khyj/0.phd_DC/DC_original_final/lung_original.csv'
synthetic_data_folder = '/home/khyj/0.phd_DC/DC_lung_final'

results_df = evaluate_synthetic_folder(real_data_path, synthetic_data_folder)
print(results_df)

# CSV로 저장
results_df.to_csv('lung_inference_risk_A_results.csv', index=False)
print("결과 저장 완료: lung_inference_risk_A_results.csv")
