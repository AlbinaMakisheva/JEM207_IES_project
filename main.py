import pandas as pd
from src.data_cleaning import clean_pfizer_data, clean_covid_data
from src.data_merging import merge_data
from src.stock_returns import calculate_stock_returns
from src.visualize_stock import visualize_stock
from src.visualize_covid_data import visualize_covid_data


covid_file_path = 'data/raw/covid_data.csv'
pfizer_file_path = 'data/raw/pfizer_stock.csv'

# Clean data
covid_data = clean_covid_data(covid_file_path)
pfizer_data = clean_pfizer_data(pfizer_file_path)

# Merge data
merged_data = merge_data(covid_data, pfizer_data)

# Calculate stock returns
merged_data = calculate_stock_returns(merged_data)

merged_data.to_csv('data/processed/merged_data.csv', index=False)

# Visualizations
visualize_stock(pfizer_data)
visualize_covid_data(covid_data)


