import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv('dcr_baseline_protection_lung_summary.csv')

# 필요한 조건만 필터링
target_conditions = ['dc_COPULAGAN', 'dc_CTGAN', 'cs_COPULAGAN', 'cs_CTGAN']
df['condition'] = df['synthetic_file'].str.extract(r'^(dc_COPULAGAN|dc_CTGAN|cs_COPULAGAN|cs_CTGAN)')

# 조건별 평균 계산
grouped = df[df['condition'].notnull()].groupby('condition').agg({
    'score': 'mean',
    'synthetic_data_median_DCR': 'mean',
    'random_data_baseline_median_DCR': 'mean'
}).reset_index()

# 결과 출력
print(grouped)

output_file = 'lung_condition_averages.csv'
grouped.to_csv(output_file, index=False)
print(f"평균값 CSV 저장 완료: {output_file}")