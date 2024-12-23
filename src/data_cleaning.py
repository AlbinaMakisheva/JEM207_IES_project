import pandas as pd

def clean_pfizer_data(file_path):
    pfizer_data = pd.read_csv(file_path, header=2)
    pfizer_data.columns = ['Date', 'Price', 'Close', 'High', 'Low', 'Volume']

    # Convert date and set index
    pfizer_data['Date'] = pd.to_datetime(pfizer_data['Date'])
    pfizer_data.set_index('Date', inplace=True)

    # drop missing values
    pfizer_data.dropna(inplace=True)  

    return pfizer_data

def clean_covid_data(file_path):
    covid_data = pd.read_csv(file_path)
    covid_data.dropna(subset=['date', 'new_cases', 'new_deaths'], inplace=True)

    # Convert date 
    covid_data['date'] = pd.to_datetime(covid_data['date'])
    
    return covid_data
