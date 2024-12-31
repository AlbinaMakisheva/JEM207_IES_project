import pandas as pd

def merge_data(covid_data, stock_data):
    return pd.merge(covid_data, stock_data, left_on='date', right_index=True, how='inner')
