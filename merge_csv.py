import pandas as pd
import os
import glob

folder_path = '/home/khyj/0. phd/dataset/DC_synthetic/no_rule/imbalanced/CTGAN/epochs-500'

# glob으로 패턴에 맞는 파일들만 선택
selected_paths = glob.glob(os.path.join(folder_path, '*5000*.csv'))
print("선택된 파일들:", selected_paths)

# 선택된 파일이 없을 때 경고
if not selected_paths:
    print("no files.")
else:
    # 각 파일 읽어서 DataFrame 리스트 만들기
    df_list = [pd.read_csv(file) for file in selected_paths]

    # 합치기
    merged_df = pd.concat(df_list, ignore_index=True)

    # 하나의 csv로 저장
    merged_df.to_csv('imbal_NoneCTGAN_ep500_merged_output.csv', index=False)
    print("saved")
