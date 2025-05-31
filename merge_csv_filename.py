import pandas as pd
import os

# csv 파일들이 들어 있는 폴더 경로
folder_path = '/home/khyj/0. phd'

# 합치고 싶은 파일 이름 리스트
selected_files = ['test.csv', 'train.csv']

# 전체 경로로 만들어주기
selected_paths = [os.path.join(folder_path, file) for file in selected_files]

# 각 선택된 파일 읽기
df_list = [pd.read_csv(file) for file in selected_paths]

# 합치기
merged_df = pd.concat(df_list, ignore_index=True)

# 하나의 csv로 저장
merged_df.to_csv('breast_original.csv', index=False)
