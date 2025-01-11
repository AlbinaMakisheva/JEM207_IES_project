import os
import numpy as np
from src.data_cleaning import clean_data
from src.data_fetching import fetch_covid_data, fetch_stock_data
from src.data_merging import merge_data

COVID_FILE = 'data/raw/covid_data.csv'
STOCK_FILE = 'data/raw/pfizer_stock.csv'

def initialize_app():
    events = {
        "WHO Declares Pandemic": "2020-03-11",
        "First Vaccine": "2020-12-08",
        "Vaccination Threshold Reached (85%)": "2021-07-30",
        "Relaxation of Lockdowns": "2022-05-01",
        "Omicron-Specific Vaccine Approval": "2022-09-01",
        "China Eases Zero-COVID Policy": "2023-01-01",
    }

    if not os.path.exists(COVID_FILE):
        fetch_covid_data(COVID_FILE)

    if not os.path.exists(STOCK_FILE):
        fetch_stock_data(STOCK_FILE)

    covid_data = clean_data(COVID_FILE)
    stock_data = clean_data(STOCK_FILE, is_stock=True)
    merged_data = merge_data(covid_data, stock_data, events)

    # Apply log transformation to selected variables
    variables_to_log_transform = ['new_deaths_smoothed', 'new_cases_smoothed']
    non_negative_data = merged_data[variables_to_log_transform]
    merged_data[variables_to_log_transform] = np.log1p(non_negative_data)

    return merged_data, events, covid_data
