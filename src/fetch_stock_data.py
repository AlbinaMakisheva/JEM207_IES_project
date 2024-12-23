import yfinance as yf
import pandas as pd

ticker = 'PFE'

stock_data = yf.download(ticker, start='2020-01-01', end='2023-12-31')

stock_data.to_csv('data/raw/pfizer_stock.csv')

print("Pfizer stock data fetched")
