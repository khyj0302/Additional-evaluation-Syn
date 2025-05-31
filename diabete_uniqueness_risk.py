import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# === 1️⃣ diabetes용 전처리 함수 ===
def preprocess_diabetes_dataset(df):
    df_processed = df.copy()
    # Age 컬럼은 구간 → 중앙값 변환
    if 'Age' in df_processed.columns:
        df_processed['Age'] = df_processed['Age'].apply(range_to_midpoint)
    # Deg_malig는 수치형 변환 (혹시 존재한다면)
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

# === 2️⃣ Uniqueness risk 계산 ===
def compute_uniqueness_risk(real_df, synthetic_df):
    match_count = 0
    for _, syn_row in synthetic_df.iterrows():
        if any((real_df == syn_row).all(axis=1)):
            match_count += 1
    return match_count / len(synthetic_df)

# === 3️⃣ 전체 폴더 순회 ===
def evaluate_synthetic_folder_uniqueness(real_path, synthetic_folder):
    real_df = pd.read_csv(real_path)
    real_df = preprocess_diabetes_dataset(real_df)

    results = []

    for file in tqdm(os.listdir(synthetic_folder)):
        if file.endswith('.csv'):
            synthetic_path = os.path.join(synthetic_folder, file)
            synthetic_df = pd.read_csv(synthetic_path)
            synthetic_df = preprocess_diabetes_dataset(synthetic_df)

            try:
                uniqueness = compute_uniqueness_risk(real_df, synthetic_df)
            except Exception as e:
                print(f"Error in {file}: {e}")
                uniqueness = np.nan

            results.append({
                'synthetic_file': file,
                'uniqueness_risk': uniqueness
            })

    results_df = pd.DataFrame(results)
    return results_df

# === 4️⃣ 실행 예시 ===
real_data_path = '/home/khyj/0.phd_DC/DC_original_final/original_diabetes.csv'
synthetic_data_folder = '/home/khyj/0.phd_DC/DC_diabetes_final'

results_df = evaluate_synthetic_folder_uniqueness(real_data_path, synthetic_data_folder)
print(results_df)

# CSV로 저장
results_df.to_csv('diabetes_uniqueness_risk_results.csv', index=False)
print("결과 저장 완료: diabetes_uniqueness_risk_results.csv")
