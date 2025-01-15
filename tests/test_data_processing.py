import pytest
import pandas as pd
from src.data_cleaning import clean_data
from src.visualization import plot_covid_cases
import matplotlib.pyplot as plt
from src.data_merging import merge_data
import os

COVID_FILE = 'data/raw/covid_data.csv'
STOCK_FILE = 'data/raw/pfizer_stock.csv'

def test_fetch_stock_data():
    assert os.path.exists(STOCK_FILE)
    stock_data = pd.read_csv(STOCK_FILE)
    assert isinstance(stock_data, pd.DataFrame)
    assert not stock_data.empty

def test_fetch_covid_data():
    assert os.path.exists(COVID_FILE)
    covid_data = pd.read_csv(COVID_FILE)
    assert isinstance(covid_data, pd.DataFrame)
    assert not covid_data.empty


def test_correct_data_types():
    data = clean_data(COVID_FILE)
    
    assert data['date'].notna().all(), "Date column contains NaN values"
    assert data['total_cases'].notna().all(), "Total cases column contains NaN values"
    assert data['new_cases'].notna().all(), "New cases column contains NaN values"


def test_plot_covid_cases():
    covid_data = pd.read_csv(COVID_FILE)
    
    covid_data['date'] = pd.to_datetime(covid_data['date'], errors='coerce')  

    required_columns = ['location', 'new_cases_smoothed', 'date']
    assert all(col in covid_data.columns for col in required_columns), "Dataset missing required columns for plotting"
    
    try:
        plot_covid_cases(covid_data)
        plt.close() 
    except Exception as e:
        pytest.fail(f"An error occurred while plotting COVID cases: {e}")
