import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata

# 평가할 synthetic CSV들이 있는 폴더
synthetic_folder = '/home/khyj/0. phd/TTGAN_dataset_final/syn_lung'
real_data_path = '/home/khyj/0. phd/TTGAN_dataset_final/original_final/original_lung.csv'

# Real 데이터 로드
real_data = pd.read_csv(real_data_path)

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

        shape_scores = []
        pair_trend_scores = []
        overall_scores = []

        for i in range(5):  # 5회 반복
            report = QualityReport()
            report.generate(real_data, synthetic_data, metadata.to_dict())

            properties = report.get_properties()
            shape_score = properties.loc[properties['Property'] == 'Column Shapes', 'Score'].values[0]
            pair_trend_score = properties.loc[properties['Property'] == 'Column Pair Trends', 'Score'].values[0]
            overall_score = report.get_score()

            shape_scores.append(shape_score)
            pair_trend_scores.append(pair_trend_score)
            overall_scores.append(overall_score)

        print(f"{file} - Shape scores: {shape_scores}, Pair Trend scores: {pair_trend_scores}, Overall scores: {overall_scores}")

        results.append({
            'Synthetic_File': file,
            'Shape_1': shape_scores[0],
            'Shape_2': shape_scores[1],
            'Shape_3': shape_scores[2],
            'Shape_4': shape_scores[3],
            'Shape_5': shape_scores[4],
            'PairTrend_1': pair_trend_scores[0],
            'PairTrend_2': pair_trend_scores[1],
            'PairTrend_3': pair_trend_scores[2],
            'PairTrend_4': pair_trend_scores[3],
            'PairTrend_5': pair_trend_scores[4],
            'Overall_1': overall_scores[0],
            'Overall_2': overall_scores[1],
            'Overall_3': overall_scores[2],
            'Overall_4': overall_scores[3],
            'Overall_5': overall_scores[4],
        })

# DataFrame으로 저장
results_df = pd.DataFrame(results)
results_df.to_csv('synthetic_quality_summary_lung_detailed.csv', index=False)
print("결과 저장 완료: synthetic_quality_summary_lung_detailed.csv")
