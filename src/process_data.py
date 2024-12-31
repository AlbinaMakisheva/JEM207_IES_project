def create_vaccination_signal(data, rate_column='new_vaccinations_smoothed_per_million', threshold=0.05):
    if rate_column not in data.columns:
        raise KeyError(f"Column '{rate_column}' not found in the dataset.")
    
    # Calculate week-over-week percentage change
    data['Vaccination_Signal'] = data[rate_column].pct_change(periods=7) > threshold
    data['Vaccination_Signal'] = data['Vaccination_Signal'].astype(int)
    return data
