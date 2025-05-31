import pandas as pd
import numpy as np
import gower
import os
from tqdm import tqdm

# === 1️⃣ 구간값 중앙값으로 변환하는 함수 ===
def range_to_midpoint(value):
    if isinstance(value, str) and '-' in value:
        low, high = map(int, value.split('-'))
        return (low + high) / 2
    else:
        try:
            return float(value)
        except:
            return np.nan

# === 2️⃣ 데이터 전처리 ===
def preprocess_dataset(df):
    df_processed = df.copy()
    for col in ['Age', 'Tumor_size', 'Inv_nodes']:
        df_processed[col] = df_processed[col].apply(range_to_midpoint)
    df_processed['Deg_malig'] = pd.to_numeric(df_processed['Deg_malig'], errors='coerce')
    return df_processed

# === 3️⃣ Uniqueness risk 계산 ===
def compute_uniqueness_risk(real_df, synthetic_df):
    match_count = 0
    for _, syn_row in synthetic_df.iterrows():
        if any((real_df == syn_row).all(axis=1)):
            match_count += 1
    return match_count / len(synthetic_df)

# === 4️⃣ Inference risk 계산 (Gower) ===
def compute_inference_risk_gower(real_df, synthetic_df):
    real_processed = preprocess_dataset(real_df)
    synthetic_processed = preprocess_dataset(synthetic_df)
    distances = gower.gower_matrix(synthetic_processed, real_processed)
    min_distances = np.min(distances, axis=1)
    return np.mean(min_distances)

# === 5️⃣ 전체 폴더 순회 ===
def evaluate_synthetic_folder(real_path, synthetic_folder):
    real_df = pd.read_csv(real_path)
    results = []

    for file in tqdm(os.listdir(synthetic_folder)):
        if file.endswith('.csv'):
            synthetic_path = os.path.join(synthetic_folder, file)
            synthetic_df = pd.read_csv(synthetic_path)

            try:
                uniqueness = compute_uniqueness_risk(real_df, synthetic_df)
                inference_risk = compute_inference_risk_gower(real_df, synthetic_df)
            except Exception as e:
                print(f"Error in {file}: {e}")
                uniqueness, inference_risk = np.nan, np.nan

            results.append({
                'synthetic_file': file,
                'uniqueness_risk': uniqueness,
                'inference_risk': inference_risk
            })

    results_df = pd.DataFrame(results)
    return results_df

# === 6️⃣ 실행 예시 ===
real_data_path = '/home/khyj/0. phd_DC/DC_original_final/breast_original.csv'
synthetic_data_folder = '/home/khyj/0. phd_DC/DC_breast_final'

results_df = evaluate_synthetic_folder(real_data_path, synthetic_data_folder)
print(results_df)

# CSV로 저장
results_df.to_csv('breast_privacy_risk_results.csv', index=False)
print("결과 저장 완료: synthetic_risk_results.csv")
