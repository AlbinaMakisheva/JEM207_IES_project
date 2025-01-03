import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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


# Prepare data for logistic regression
def prepare_binary_target(data, price_column='close'):
    data['price_change'] = (data[price_column].diff() > 0).astype(int)
    return data

# Perform logistic regression
def perform_logistic_regression(data, independent_vars, target_var='price_change'):
    if target_var not in data.columns or not all(var in data.columns for var in independent_vars):
        missing = [var for var in [target_var] + independent_vars if var not in data.columns]
        raise KeyError(f"Missing columns: {', '.join(missing)} in the dataset.")

    regression_data = data[independent_vars + [target_var]].dropna()
    X = regression_data[independent_vars]
    y = regression_data[target_var]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 30% of the dataset will be used for testing, and the remaining 70% will be used for training
    
    # Fit logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Model evaluation
    print("Model Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print("Accuracy on Test Data:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model