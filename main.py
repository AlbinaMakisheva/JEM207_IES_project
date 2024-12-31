import os
from src.data_cleaning import clean_data
from src.data_merging import merge_data
from src.analysis import compute_daily_returns, perform_regression_analysis
from src.visualization import plot_covid_cases, plot_stock_with_events
from src.data_fetching import fetch_covid_data, fetch_stock_data

COVID_FILE = 'data/raw/covid_data.csv'
STOCK_FILE = 'data/raw/pfizer_stock.csv'

def main():
    if not os.path.exists(COVID_FILE):
        print("Fetching COVID data...")
        fetch_covid_data(COVID_FILE)
        
    if not os.path.exists(STOCK_FILE):
        print("Fetching stock data...")
        fetch_stock_data(STOCK_FILE)

    # Clean data
    covid_data = clean_data(COVID_FILE)
    stock_data = clean_data(STOCK_FILE, is_stock=True)
    
    # Merge datasets
    merged_data = merge_data(covid_data, stock_data)

    # Analyze and visualize
    merged_data = compute_daily_returns(merged_data)
    regression_model = perform_regression_analysis(merged_data, 'new_cases', 'daily_return')
    
    # Key events
    events = {
        "WHO Declares Pandemic": "2020-03-11",
        "First Vaccine": "2020-12-08",
        "Vaccination Threshold Reached (85%)": "2021-07-30",
    }
    plot_stock_with_events(merged_data, events)
    plot_covid_cases(merged_data)

if __name__ == "__main__":
    main()
