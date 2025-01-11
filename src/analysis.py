import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import numpy as np

# Filter data around key events
def filter_data_around_events(data, events, window_months=1, date_column='date'):
    event_dates = pd.to_datetime(list(events.values()))
    filtered_data = pd.DataFrame()
    
    window_months = int(window_months)
    
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
    
    return model, r2_score

# Analyze the impact of events on stock returns
def analyze_event_impact(merged_data, events, window_size):
    results = {}
    for event, event_date in events.items():
        event_window = (merged_data['date'] >= pd.to_datetime(event_date) - pd.DateOffset(months=window_size)) & \
                       (merged_data['date'] <= pd.to_datetime(event_date) + pd.DateOffset(months=window_size))
        event_data = merged_data[event_window]
        X = event_data[['new_vaccinations_smoothed', 'new_deaths_smoothed']]
        y = event_data['daily_return']
        model = LinearRegression().fit(X, y)
        results[event] = model.score(X, y)
    return results


def prepare_binary_target(df, price_column='close'):
    df['price_change'] = (df[price_column].diff() > 0).astype(int)
    df['target'] = (df[price_column].shift(-1) > df[price_column]).astype(int)
    return df


# Perform logistic regression
def perform_logistic_regression(data, independent_vars, target_var='target'):
    if target_var not in data.columns or not all(var in data.columns for var in independent_vars):
        missing = [var for var in [target_var] + independent_vars if var not in data.columns]
        raise KeyError(f"Missing columns: {', '.join(missing)} in the dataset.")

    regression_data = data[independent_vars + [target_var]].dropna()
    X = regression_data[independent_vars]
    y = regression_data[target_var]
    
    # Split data into 2 sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
    
    # Fit logistic  model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    return model, accuracy, fpr, tpr, roc_auc

def perform_extended_logistic_regression(data, independent_vars, target_var='target'):
    if target_var not in data.columns or not all(var in data.columns for var in independent_vars):
        missing = [var for var in [target_var] + independent_vars if var not in data.columns]
        raise KeyError(f"Missing columns: {', '.join(missing)} in the dataset.")

    # Prepare data for training
    log_data = data[independent_vars + [target_var]].dropna()
    X = log_data[independent_vars]
    y = log_data[target_var]
    
    # Split data into 2 sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit Logistic Regression model
    log_model = LogisticRegression(max_iter=10000)
    log_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = log_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, log_model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    
    return log_model, accuracy, fpr, tpr, roc_auc, X_test, y_test


# Perform Random Forest classification
def perform_random_forest(data, independent_vars, target_var='target'):
    if target_var not in data.columns or not all(var in data.columns for var in independent_vars):
        missing = [var for var in [target_var] + independent_vars if var not in data.columns]
        raise KeyError(f"Missing columns: {', '.join(missing)} in the dataset.")

    # Prepare data for training
    rf_data = data[independent_vars + [target_var]].dropna()
    X = rf_data[independent_vars]
    y = rf_data[target_var]
    
    # Split data into 2 sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    
    return rf_model, accuracy, fpr, tpr, roc_auc


# Function to display the results in Streamlit
def display_analysis_results(regression_model, r2_score, event_impact, rf_model):
    st.subheader("Regression Results")
    st.write(f"R²: {r2_score:.4f}")
    st.write("Coefficients:")
    st.write(dict(zip(regression_model.feature_names_in_, regression_model.coef_)))
    st.write(f"Intercept: {regression_model.intercept_:.4f}")
    
    st.subheader("Event Impact Analysis")
    st.write(event_impact)
    
    st.subheader("Random Forest Feature Importance")
    st.write(rf_model.feature_importances_)
