import os
import pandas as pd
from tqdm import tqdm
from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata


# 평가할 synthetic CSV들이 있는 폴더
synthetic_folder = '/home/khyj/0. phd/TTGAN_dataset_final/syn_liver'  # ← 이 경로 바꿔줘
real_data_path = '/home/khyj/0. phd/TTGAN_dataset_final/original_final/original_liver.csv'    # ← 진짜 원본 데이터 경로


# Real 데이터 로드
real_data = pd.read_csv(real_data_path)

# 결과 저장 리스트
results = []

# 메타데이터 자동 감지
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)

# 결과 저장 리스트
results = []

# 파일 순회
for file in tqdm(os.listdir(synthetic_folder)):
    if file.endswith('.csv'):
        synthetic_path = os.path.join(synthetic_folder, file)
        synthetic_data = pd.read_csv(synthetic_path)

        # QualityReport 생성
        report = QualityReport()
        report.generate(real_data, synthetic_data, metadata.to_dict())

        # 개별 속성 추출
        properties = report.get_properties()
        shape_score = properties.loc[properties['Property'] == 'Column Shapes', 'Score'].values[0]
        pair_trend_score = properties.loc[properties['Property'] == 'Column Pair Trends', 'Score'].values[0]
        overall_score = report.get_score()

        print(f"{file} - Shape: {shape_score:.4f}, Pair Trend: {pair_trend_score:.4f}, Overall: {overall_score:.4f}")

        results.append({
            'Synthetic_File': file,
            'Shape': round(shape_score, 4),
            'Pair_Trend': round(pair_trend_score, 4),
            'Overall': round(overall_score, 4)
        })

# DataFrame으로 저장
results_df = pd.DataFrame(results)
results_df.to_csv('synthetic_quality_summary_liver.csv', index=False)
print("결과 저장 완료: synthetic_quality_summary_liver.csv")
