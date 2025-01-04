import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Filter data around key events
def filter_data_around_events(data, events, window_months=1, date_column='date'):
    event_dates = pd.to_datetime(list(events.values()))
    filtered_data = pd.DataFrame()
    
    for event_date in event_dates:
        start_date = event_date - pd.DateOffset(months=window_months)
        end_date = event_date + pd.DateOffset(months=window_months)
        event_data = data[(data[date_column] >= start_date) & (data[date_column] <= end_date)]
        filtered_data = pd.concat([filtered_data, event_data], ignore_index=True)
    
    return filtered_data

# Perform regression analysis
def perform_multiple_linear_regression(data, dependent_var, independent_vars):
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