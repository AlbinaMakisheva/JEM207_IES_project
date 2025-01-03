import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Calculates daily returns for stock prices
def compute_daily_returns(data, price_column='close'):
    data['daily_return'] = data[price_column].pct_change()
    return data


def perform_multiple_linear_regression(data, dependent_var, independent_vars):
    independent_vars = [
        'new_vaccinations_smoothed_per_million',
        'new_cases',
        'Dummy_Variable',
        'stringency_index',
        'new_cases_per_million',
        'total_vaccinations_per_hundred',
        'positive_rate',
        'gdp_per_capita',
        'reproduction_rate'
    ]
    if any(var not in data.columns for var in independent_vars) or dependent_var not in data.columns:
        raise KeyError("Missing required columns for regression.")
    regression_data = data[[dependent_var] + independent_vars].dropna()
    X = regression_data[independent_vars]
    y = regression_data[dependent_var]
    
    model = LinearRegression()
    model.fit(X, y)
    
    r2_score = model.score(X, y)
    
    print(f"Coefficients: {dict(zip(independent_vars, model.coef_))}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"R^2: {r2_score:.4f}")
    return model, r2_score


# Analyze the impact of events on stock returns
def analyze_event_impact(data, event_column='Dummy_Variable', return_column='daily_return'):
    if event_column not in data.columns or return_column not in data.columns:
        raise KeyError(f"Columns '{event_column}' or '{return_column}' not found in the dataset.")
    
    # Group by event occurrence and calculate mean returns
    event_impact = data.groupby(event_column)[return_column].mean().reset_index()
    print("Event Impact Analysis:\n", event_impact)
    return event_impact
