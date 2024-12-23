import pandas as pd

def calculate_stock_returns(merged_data):
    merged_data['Stock_Returns'] = merged_data['Close'].pct_change()

    return merged_data
