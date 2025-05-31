import pandas as pd
import json
import os
import glob
from sdv.metadata import SingleTableMetadata
from sdmetrics.single_table import DCRBaselineProtection

# 1️ 원본(real) 데이터 불러오기
real_table = pd.read_csv('/home/khyj/0. phd/TTGAN_dataset_final/original_final/original_liver.csv')

# 2️ SingleTableMetadata 생성 및 자동 탐지
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_table)

# 3️ synthetic CSV 파일들이 들어 있는 폴더
synthetic_folder = '/home/khyj/0. phd/TTGAN_dataset_final/syn_liver'
synthetic_files = glob.glob(os.path.join(synthetic_folder, '*.csv'))

print(f"총 {len(synthetic_files)}개의 synthetic 파일을 처리합니다.")

# 4️ 결과 담을 리스트
results = []

# 5️ 각 synthetic 파일에 대해 반복
for file_path in synthetic_files:
    synthetic_table = pd.read_csv(file_path)
    
    score = DCRBaselineProtection.compute_breakdown(
        real_data=real_table,
        synthetic_data=synthetic_table,
        metadata=metadata.to_dict()
    )
    
    # 결과에 파일 이름 추가
    flat_result = {
        'synthetic_file': os.path.basename(file_path),
        'score': score.get('score'),
        'synthetic_data_median_DCR': score['median_DCR_to_real_data'].get('synthetic_data'),
        'random_data_baseline_median_DCR': score['median_DCR_to_real_data'].get('random_data_baseline')
    }
    
    results.append(flat_result)
    print(f" {os.path.basename(file_path)} 처리 완료.")

# 6️ 전체 결과 DataFrame으로 만들기
results_df = pd.DataFrame(results)

# 7️ CSV로 저장
output_csv = 'dcr_baseline_protection_liver_summary.csv'
results_df.to_csv(output_csv, index=False)
print(f"전체 결과 CSV 저장 완료: {output_csv}")
