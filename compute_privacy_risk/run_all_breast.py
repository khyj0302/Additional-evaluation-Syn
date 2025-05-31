import pandas as pd
from compute_uniqueness_risk import compute_uniqueness_risk
from compute_cap import compute_cap
from compute_inference_risk import compute_inference_risk
import os

# 데이터 로드
real_df = pd.read_csv('/home/khyj/0. phd_DC/DC_original_final/breast_original.csv')

synthetic_folder = '/home/khyj/0. phd_DC/DC_breast_final'


quasi_ids = ['Age', 'Menopause', 'Breast', 'Breast_quad']
sensitive_col = 'Class'
num_columns = ['Tumor_size', 'Inv_nodes', 'Deg_malig']

# 결과 담을 리스트
results = []

# synthetic 폴더 내 모든 CSV 파일 반복
for filename in os.listdir(synthetic_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(synthetic_folder, filename)
        synthetic_df = pd.read_csv(file_path)

        # 각 위험지표 계산
        print(f"Processing: {filename}")
        uniqueness = compute_uniqueness_risk(real_df, synthetic_df)
        cap = compute_cap(real_df, synthetic_df, quasi_ids, sensitive_col)
        inference_risk = compute_inference_risk(real_df, synthetic_df, num_columns)

        # 결과 저장
        results.append({
            'file': filename,
            'uniqueness_risk': uniqueness,
            'linkage_cap': cap,
            'inference_risk': inference_risk
        })

# 결과 DataFrame으로 정리
results_df = pd.DataFrame(results)

# CSV로 저장
output_file = 'risk_summary.csv'
results_df.to_csv(output_file, index=False)
print(f"\n✅ All results saved to {output_file}")
