import requests
import pandas as pd

url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"

covid_data = pd.read_csv(url)

covid_data.to_csv('data/raw/covid_data.csv', index=False)

print("COVID data fetched")
