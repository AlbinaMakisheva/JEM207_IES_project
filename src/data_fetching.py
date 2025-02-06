import yfinance as yf
import pandas as pd
import os

def fetch_stock_data():
    ticker = 'PFE'
    file_path = 'data/raw/pfizer_stock.csv'

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        stock_data = yf.download(ticker, start='2020-01-01', end='2023-12-29')
        stock_data.to_csv(file_path)
        print("Pfizer stock data fetched")
    except Exception as e:
        print(f"Error fetching stock data: {e}")


def fetch_covid_data():
    url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    file_path = 'data/raw/covid_data.csv'

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        covid_data = pd.read_csv(url)
        covid_data.to_csv(file_path, index=False)
        print("COVID data fetched")
    except Exception as e:
        print(f"Error fetching COVID data: {e}")
