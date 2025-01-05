import pandas as pd

def clean_data(file_path, is_stock=False, rate_column='new_vaccinations_smoothed_per_million', threshold=0.05):
    if is_stock:
        data = pd.read_csv(file_path, header=2)
        data.columns = ['date', 'price', 'close', 'high', 'low', 'volume']
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        data['daily_return'] = data['close'].pct_change()
    else:
        data = pd.read_csv(file_path)
        # Ensure existence
        if not {'date', 'new_cases', 'new_deaths'}.issubset(data.columns):
            raise KeyError("Missing required columns: 'date', 'new_cases', or 'new_deaths'")
        
        data.dropna(subset=['date', 'new_cases', 'new_deaths'], inplace=True)
        data['date'] = pd.to_datetime(data['date'])
        
        # Apply vaccination signal
        data = create_vaccination_signal(data, rate_column, threshold)

    # Standardize column names
    data.columns = data.columns.str.lower()

    return data

def create_vaccination_signal(data, rate_column='new_vaccinations_smoothed_per_million', threshold=0.05):
    if rate_column not in data.columns:
        raise KeyError(f"Column '{rate_column}' not found in the dataset.")
    
    # Calculate week-over-week percentage change
    data['vaccination_signal'] = (data[rate_column].pct_change(periods=7, fill_method=None) > threshold).astype(int)
    
    return data

