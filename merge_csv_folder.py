import pandas as pd
import glob
import os

# csv 파일들이 들어 있는 폴더 경로
folder_path = '/home/khyj/0. phd/TTGAN_dataset_final/original_lung'

# 폴더 안의 모든 csv 파일 경로 리스트
all_csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# 모든 csv 파일 읽어서 하나의 DataFrame으로 합치기
df_list = [pd.read_csv(file) for file in all_csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# 결과를 하나의 csv로 저장 (선택)
merged_df.to_csv('original_lung.csv', index=False)
