import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

def convert_range_to_midpoint(series):
    """
    예: '30-34' → 32, '5-9' → 7
    """
    numeric = series.str.extract(r'(\d+)-(\d+)').astype(float)
    midpoint = numeric.mean(axis=1)
    return midpoint

def preprocess_numeric_columns(df, columns):
    """
    DataFrame에서 지정한 columns을 숫자(float)로 변환.
    구간형 문자열은 중앙값으로, 숫자형은 그대로.
    """
    df_processed = df.copy()
    for col in columns:
        if df_processed[col].dtype == 'object':
            df_processed[col] = convert_range_to_midpoint(df_processed[col])
        else:
            df_processed[col] = df_processed[col].astype(float)
    return df_processed[columns].to_numpy()

def compute_inference_risk(real_df, synthetic_df, columns):
    real_matrix = preprocess_numeric_columns(real_df, columns)
    synthetic_matrix = preprocess_numeric_columns(synthetic_df, columns)
    
    distances = pairwise_distances(synthetic_matrix, real_matrix, metric='euclidean')
    min_distances = np.min(distances, axis=1)
    average_min_distance = np.mean(min_distances)
    
    return average_min_distance
