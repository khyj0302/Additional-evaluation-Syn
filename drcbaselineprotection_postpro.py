import pandas as pd
import numpy as np
import json
import os
import glob
from sdv.metadata import SingleTableMetadata
from sdmetrics.single_table import DCRBaselineProtection

# 1️⃣ 원본(real) 데이터 불러오기
real_table = pd.read_csv('/home/khyj/0. phd_DC/DC_original_final/breast_original.csv')

# 2️⃣ SingleTableMetadata 생성 및 자동 탐지
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_table)

# numeric / categorical 컬럼 자동 추출
numeric_cols = real_table.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = real_table.select_dtypes(include=['object']).columns

# 3️⃣ synthetic CSV 파일들이 들어 있는 폴더
synthetic_folder = '/home/khyj/0. phd_DC/DC_breast_final'
synthetic_files = glob.glob(os.path.join(synthetic_folder, '*.csv'))

print(f"총 {len(synthetic_files)}개의 synthetic 파일을 처리합니다.")

# 4️⃣ 결과 담을 리스트
results = []

# 5️⃣ 각 synthetic 파일에 대해 반복
for file_path in synthetic_files:
    synthetic_table = pd.read_csv(file_path)
    
    # === ✅ POST-PROCESSING START ===
    # numeric noise 추가
    for col in numeric_cols:
        synthetic_table[col] += np.random.normal(0, 0.05 * synthetic_table[col].std(), size=synthetic_table.shape[0])
    
    # rare class masking
    threshold = 0.01 * len(synthetic_table)
    for col in categorical_cols:
        rare_classes = synthetic_table[col].value_counts()[synthetic_table[col].value_counts() < threshold].index
        synthetic_table[col] = synthetic_table[col].apply(lambda x: 'Other' if x in rare_classes else x)
    
    # outlier clipping
    for col in numeric_cols:
        lower = synthetic_table[col].quantile(0.01)
        upper = synthetic_table[col].quantile(0.99)
        synthetic_table[col] = np.clip(synthetic_table[col], lower, upper)
    
    # shuffle
    synthetic_table = synthetic_table.sample(frac=1).reset_index(drop=True)
    # === ✅ POST-PROCESSING END ===
    
    # DCR Baseline Protection Score 계산
    score = DCRBaselineProtection.compute_breakdown(
        real_data=real_table,
        synthetic_data=synthetic_table,
        metadata=metadata.to_dict()
    )
    
    flat_result = {
        'synthetic_file': os.path.basename(file_path),
        'score': score.get('score'),
        'synthetic_data_median_DCR': score['median_DCR_to_real_data'].get('synthetic_data'),
        'random_data_baseline_median_DCR': score['median_DCR_to_real_data'].get('random_data_baseline')
    }
    
    results.append(flat_result)
    print(f"✅ {os.path.basename(file_path)} 처리 완료.")

# 6️⃣ 전체 결과 DataFrame으로 만들기
results_df = pd.DataFrame(results)

# 7️⃣ CSV로 저장
output_csv = 'dcr_baseline_protection_breast_summary_postprocessed.csv'
results_df.to_csv(output_csv, index=False)
print(f"✅ 전체 결과 CSV 저장 완료: {output_csv}")
