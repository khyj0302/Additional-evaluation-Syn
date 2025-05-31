import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv('/home/khyj/0. phd_DC/drc_baseline_protection_result/dcr_baseline_protection_lung_summary.csv')

# 조건 컬럼 만들기 (앞부분 접두사 추출)
df['condition'] = df['synthetic_file'].str.extract(r'^(imbal_CSCTGAN|imbal_DCCTGAN|imbal_CSCOPULAGAN|imbal_DCCOPULAGAN)')

# 필요한 조건만 필터링
filtered_df = df[df['condition'].notnull()]

# 조건별 평균 계산
grouped = filtered_df.groupby('condition').agg({
    'score': 'mean',
    'synthetic_data_median_DCR': 'mean',
    'random_data_baseline_median_DCR': 'mean'
}).reset_index()

# 결과 출력
print("조건별 평균값:")
print(grouped)

# CSV로 저장
output_file = 'lung_balanced_condition_averages_new.csv'
grouped.to_csv(output_file, index=False)
print(f"✅ 평균값 CSV 저장 완료: {output_file}")
