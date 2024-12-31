import pandas as pd
from src.data_cleaning import clean_pfizer_data, clean_covid_data, clean_merged_dummy_data
from src.data_merging import merge_data
from src.stock_returns import calculate_stock_returns
from src.visualize_stock import visualize_stock
from src.visualize_covid_data import visualize_covid_data
from src.visualize_with_dummy import plot_stock_price
from src.visualize_with_dummy import create_dummy_variable
from src.visualize_cases_stocks import plot_cases_vs_stock
from src.visualize_cases_country import plot_cases_country
from src.process_data import create_vaccination_signal


import os

# Create directory if it doesn't exist
os.makedirs('data/processed', exist_ok=True)


covid_file_path = 'data/raw/covid_data.csv'
pfizer_file_path = 'data/raw/pfizer_stock.csv'

merged_data_with_dummy = pd.read_csv('data/processed/merged_data_with_dummy.csv')
merged_data_with_dummy['date'] = pd.to_datetime(merged_data_with_dummy['date'])

# Clean data
covid_data = clean_covid_data(covid_file_path)
pfizer_data = clean_pfizer_data(pfizer_file_path)
merged_data_with_dummy = clean_merged_dummy_data(merged_data_with_dummy)

# Merge data
merged_data = merge_data(covid_data, pfizer_data)

# Calculate stock returns
merged_data = calculate_stock_returns(merged_data)

# Visualizations
# visualize_stock(pfizer_data)
# visualize_covid_data(covid_data)
# plot_stock_price()
# plot_cases_vs_stock(merged_data_with_dummy)
plot_cases_country(merged_data)

vaccination_signal_data = create_vaccination_signal(
    data=merged_data_with_dummy,
    rate_column='new_vaccinations_smoothed_per_million'
)

