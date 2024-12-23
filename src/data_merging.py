import pandas as pd

def merge_data(covid_data, pfizer_data):
    merged_data = pd.merge(covid_data, pfizer_data, left_on='date', right_index=True, how='inner')

    return merged_data
