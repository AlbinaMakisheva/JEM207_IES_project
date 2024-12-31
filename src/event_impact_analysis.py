import pandas as pd

def compute_daily_returns(data, price_column='Close'):
    if price_column not in data.columns:
        raise KeyError(f"Column '{price_column}' not found in the dataset.")
    data['daily_return'] = data[price_column].pct_change()
    return data

def analyze_event_impact(data, event_column='Dummy_Variable', return_column='daily_return'):
    if event_column not in data.columns or return_column not in data.columns:
        raise KeyError(f"Columns '{event_column}' or '{return_column}' not found in the dataset.")
    
    # Group by event occurrence and calculate mean returns
    event_impact = data.groupby(event_column)[return_column].mean().reset_index()
    print("Event Impact Analysis:\n", event_impact)
    return event_impact
