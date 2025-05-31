import pandas as pd

def compute_uniqueness_risk(real_df, synthetic_df):
    match_count = 0
    for _, syn_row in synthetic_df.iterrows():
        if any((real_df == syn_row).all(axis=1)):
            match_count += 1
    uniqueness_risk = match_count / len(synthetic_df)
    return uniqueness_risk
