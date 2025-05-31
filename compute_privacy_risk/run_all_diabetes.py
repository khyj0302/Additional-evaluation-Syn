import pandas as pd
import os
from compute_uniqueness_risk import compute_uniqueness_risk
from compute_cap import compute_cap
from compute_inference_risk_diabete import compute_inference_risk

# =========================
# ⚙️ 설정
# =========================
real_df = pd.read_csv('/home/khyj/0. phd_DC/DC_original_final/diabetes_original.csv')
synthetic_folder = '/home/khyj/0. phd_DC/DC_diabetes_final'

# ⚠️ 컬럼 분류
quasi_ids = ['Race', 'Gender', 'Age', 'Medical_specialty']
sensitive_col = 'Readmitted'
num_columns = ['Time_in_hospital', 'Num_lab_procedures', 'Num_procedures',
               'Num_medications', 'Number_outpatient', 'Number_emergency',
               'Number_inpatient', 'Number_diagnoses']

# Age 컬럼 숫자 추출 ('age_7.0' → 7.0)
def extract_numeric_part(series):
    return series.str.extract(r'(\d+\.\d+)').astype(float)[0]

if 'Age' in real_df.columns:
    real_df['Age'] = extract_numeric_part(real_df['Age'])

# =========================
# 📊 결과 담을 리스트
# =========================
results = []

# =========================
# 📁 synthetic 폴더 내 모든 CSV 반복
# =========================
for filename in os.listdir(synthetic_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(synthetic_folder, filename)
        synthetic_df = pd.read_csv(file_path)

        # Age 컬럼 숫자 추출
        if 'Age' in synthetic_df.columns:
            synthetic_df['Age'] = extract_numeric_part(synthetic_df['Age'])

        print(f"Processing: {filename}")
        try:
            uniqueness = compute_uniqueness_risk(real_df, synthetic_df)
            cap = compute_cap(real_df, synthetic_df, quasi_ids, sensitive_col)
            inference_risk = compute_inference_risk(real_df, synthetic_df, num_columns)

            results.append({
                'file': filename,
                'uniqueness_risk': uniqueness,
                'linkage_cap': cap,
                'inference_risk': inference_risk
            })
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

# =========================
# 💾 결과 저장
# =========================
results_df = pd.DataFrame(results)
output_file = 'risk_summary_diabetes.csv'
results_df.to_csv(output_file, index=False)
print(f"\n✅ All results saved to {output_file}")
