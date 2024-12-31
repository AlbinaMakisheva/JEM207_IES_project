import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Calculates daily returns for stock prices
def compute_daily_returns(data, price_column='close'):
    data['daily_return'] = data[price_column].pct_change()
    return data


# Regression analysis between two variables
def perform_regression_analysis(data, independent_var, dependent_var):
    if independent_var not in data.columns or dependent_var not in data.columns:
        raise KeyError(f"Columns '{independent_var}' or '{dependent_var}' not found in the dataset.")
    
    regression_data = data[[independent_var, dependent_var]].dropna()
    X = regression_data[independent_var].values.reshape(-1, 1)
    y = regression_data[dependent_var].values

    model = LinearRegression()
    model.fit(X, y)

    print(f"Coefficient: {model.coef_[0]:.4f}, Intercept: {model.intercept_:.4f}, R^2: {model.score(X, y):.4f}")
    return model

# Analyze the impact of events on stock returns
def analyze_event_impact(data, event_column='Dummy_Variable', return_column='daily_return'):
    if event_column not in data.columns or return_column not in data.columns:
        raise KeyError(f"Columns '{event_column}' or '{return_column}' not found in the dataset.")
    
    # Group by event occurrence and calculate mean returns
    event_impact = data.groupby(event_column)[return_column].mean().reset_index()
    print("Event Impact Analysis:\n", event_impact)
    return event_impact
