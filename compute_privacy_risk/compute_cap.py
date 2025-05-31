import pandas as pd

def compute_cap(real_df, synthetic_df, quasi_identifiers, sensitive_column):
    correct_matches = 0
    for _, real_row in real_df.iterrows():
        qi_values = real_row[quasi_identifiers]
        synthetic_matches = synthetic_df[synthetic_df[quasi_identifiers].eq(qi_values).all(axis=1)]
        if not synthetic_matches.empty:
            if any(synthetic_matches[sensitive_column] == real_row[sensitive_column]):
                correct_matches += 1
    cap = correct_matches / len(real_df)
    return cap
