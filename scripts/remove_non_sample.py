import pandas as pd

def remove_non_samples(df):
    df = df[df['PrimarySample'] == True]
    return df