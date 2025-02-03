import pandas as pd

def clean_data(file_path, is_stock=False, rate_column='new_vaccinations_smoothed_per_million', threshold=0.05):
    if is_stock:
        data = pd.read_csv(file_path, header=2)
        data = data.iloc[:, :6] 
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
        
        # Handle missing or zero values in new_cases and new_deaths
        data = handle_missing_or_zero_values(data, ['new_cases', 'new_deaths'])

        # Apply vaccination signal
        data = create_vaccination_signal(data, rate_column, threshold)

    # Standardize column names
    data.columns = data.columns.str.lower()

    return data

def handle_missing_or_zero_values(data, columns):
    """
    Handles missing or zero values in specified columns.
    Missing values are filled with the mean of the column.
    Zero values are treated as missing and filled similarly.
    """
    for col in columns:
        data[col] = data[col].replace(0, pd.NA)  # Replace zeros with NaN
        data[col] = data[col].fillna(data[col].mean())  # Fill NaN with column mean
    return data

def create_vaccination_signal(data, rate_column='new_vaccinations_smoothed_per_million', threshold=0.05):
    if rate_column not in data.columns:
        raise KeyError(f"Column '{rate_column}' not found in the dataset.")
    
    # Handle missing or zero values in the rate_column
    data[rate_column] = data[rate_column].replace(0, pd.NA)
    data[rate_column] = data[rate_column].fillna(data[rate_column].mean())

    # Calculate week-over-week percentage change
    data['vaccination_signal'] = (data[rate_column].pct_change(periods=7, fill_method=None) > threshold).astype(int)
    
    return data

