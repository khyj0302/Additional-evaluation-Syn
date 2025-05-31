import pandas as pd
import numpy as np
import gower
import os
from tqdm import tqdm

# === 1️⃣ diabetes용 전처리 함수 ===
def preprocess_diabetes_dataset(df):
    df_processed = df.copy()
    # Age만 구간 문자열 → 중앙값 변환
    if 'Age' in df_processed.columns:
        df_processed['Age'] = df_processed['Age'].apply(range_to_midpoint)
    # Deg_malig는 수치형 처리 (혹시 존재하면)
    if 'Deg_malig' in df_processed.columns:
        df_processed['Deg_malig'] = pd.to_numeric(df_processed['Deg_malig'], errors='coerce')
    # Class 컬럼 제거 (optional)
    if 'Class' in df_processed.columns:
        df_processed = df_processed.drop(columns=['Class'])
    return df_processed

def range_to_midpoint(value):
    if isinstance(value, str) and '-' in value:
        try:
            low, high = map(int, value.split('-'))
            return (low + high) / 2
        except:
            return np.nan
    else:
        try:
            return float(value)
        except:
            return np.nan

# === 2️⃣ real–real 최소 거리 계산 ===
def compute_real_min_distances(real_df):
    real_matrix = gower.gower_matrix(real_df)
    np.fill_diagonal(real_matrix, np.inf)
    return np.min(real_matrix, axis=1)

# === 3️⃣ synthetic–real 최소 거리 계산 ===
def compute_synthetic_min_distances(synthetic_df, real_df):
    syn_real_matrix = gower.gower_matrix(synthetic_df, real_df)
    return np.min(syn_real_matrix, axis=1)

# === 4️⃣ 추론 위험도 A 계산 (평균 기준) ===
def compute_inference_risk_A(real_df, synthetic_df):
    d_o = compute_real_min_distances(real_df)
    d_s = compute_synthetic_min_distances(synthetic_df, real_df)
    d_o_mean = np.mean(d_o)
    A = np.mean(d_s < d_o_mean)
    return A

# === 5️⃣ 폴더 순회 ===
def evaluate_synthetic_folder(real_path, synthetic_folder):
    real_df = pd.read_csv(real_path)
    real_df = preprocess_diabetes_dataset(real_df)

    results = []
    for file in tqdm(os.listdir(synthetic_folder)):
        if file.endswith('.csv'):
            syn_path = os.path.join(synthetic_folder, file)
            synthetic_df = pd.read_csv(syn_path)
            synthetic_df = preprocess_diabetes_dataset(synthetic_df)

            try:
                risk_score = compute_inference_risk_A(real_df, synthetic_df)
            except Exception as e:
                print(f"Error in {file}: {e}")
                risk_score = np.nan

            results.append({
                'synthetic_file': file,
                'inference_risk_A': risk_score
            })

    return pd.DataFrame(results)

# === 6️⃣ 실행 예시 ===
real_data_path = '/home/khyj/0. phd_DC/DC_original_final/diabetes_original.csv'
synthetic_data_folder = '/home/khyj/0. phd_DC/DC_diabetes_final'

results_df = evaluate_synthetic_folder(real_data_path, synthetic_data_folder)
results_df.to_csv('diabetes_inference_risk_A_mean_based.csv', index=False)
print("완료! 결과 저장: diabetes_inference_risk_A_mean_based.csv")
