import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .data_merging import merge_data


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


#Testing autocorrelation between variables

def plot_lag_correlations(data, lag=1):
    """
    Plots lag correlations for all numeric columns in the DataFrame.
    
    Parameters:
    - data (pd.DataFrame): The input DataFrame with numeric columns.
    - lag (int): The lag to use for correlation analysis.
    
    Raises:
    - ValueError: If `data` contains no numeric columns.
    """
    # Ensure `data` is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"Expected a Pandas DataFrame, but got {type(data)}")

    # Filter numeric columns only
    numeric_data = data.select_dtypes(include=['number'])

    # Check if there are numeric columns to process
    if numeric_data.empty:
        raise ValueError("The DataFrame contains no numeric columns for analysis.")
    
    # Create lagged DataFrame
    lagged_data = numeric_data.shift(lag)
    
    # Compute correlation matrix between original and lagged data
    corr_matrix = numeric_data.corrwith(lagged_data, axis=0)
    st.write(f"Autocorrelation at lag {lag} for all variables:")
    st.dataframe(corr_matrix)  # Display as a DataFrame in Streamlit
    
    # Visualize lag plots for all numeric variables
    for column in numeric_data.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        pd.plotting.lag_plot(numeric_data[column], lag=lag, ax=ax)
        ax.set_title(f'Lag-{lag} Autocorrelation for {column}')
        ax.set_xlabel(f'{column} (t)')
        ax.set_ylabel(f'{column} (t-{lag})')
        st.pyplot(fig) 

#Perfoming Stationarity: Augmented Dickey-Fuller Test

def check_stationarity(data):
    # Filter numeric columns
    numeric_data = data.select_dtypes(include=['number'])
    
    results = []
    for column in numeric_data.columns:
        adf_result = adfuller(numeric_data[column].dropna())
        results.append({
            'Variable': column,
            'ADF Statistic': adf_result[0],
            'p-value': adf_result[1]
        })
    
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    
    st.write("Stationarity Test Results (ADF Test):")
    st.table(results_df)
    
    return results_df

# Reduce Multicollinearity: Variance Inflation Factor (VIF)

def reduce_multicollinearity(data, threshold=10):
    """
    Checks for multicollinearity using Variance Inflation Factor (VIF) and identifies problematic variables.
    """
    # Filter numeric columns
    numeric_data = data.select_dtypes(include=['number']).dropna()
    
    # Compute VIF for each numeric column
    vif_data = pd.DataFrame()
    vif_data['Feature'] = numeric_data.columns
    vif_data['VIF'] = [
        variance_inflation_factor(numeric_data.values, i) 
        for i in range(numeric_data.shape[1])
    ]
    
    st.write("Multicollinearity Analysis (VIF):")
    st.table(vif_data)
    
    # Identify features with high VIF
    high_vif_features = vif_data[vif_data['VIF'] > threshold]
    if not high_vif_features.empty:
        st.warning("Features with high VIF (potential multicollinearity):")
        st.table(high_vif_features)
    else:
        st.success("No multicollinearity detected (VIF below threshold).")
    
    return vif_data

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
def analyze_event_impact(data, event_column='Dummy_Variable', return_column='daily_return'):
    if event_column not in data.columns or return_column not in data.columns:
        raise KeyError(f"Columns '{event_column}' or '{return_column}' not found in the dataset.")
    
    # Group by event occurrence and calculate mean returns
    event_impact = data.groupby(event_column)[return_column].mean().reset_index()
    print("Event Impact Analysis:\n", event_impact)
    return event_impact


def prepare_binary_target(df, price_column='close'):
    df['price_change'] = (df[price_column].diff() > 0).astype(int)
    df['target'] = (df[price_column].shift(-1) > df[price_column]).astype(int)
    return df


# Perform logistic regression
def perform_logistic_regression(data, independent_vars, target_var='price_change'):
    if target_var not in data.columns or not all(var in data.columns for var in independent_vars):
        missing = [var for var in [target_var] + independent_vars if var not in data.columns]
        raise KeyError(f"Missing columns: {', '.join(missing)} in the dataset.")

    regression_data = data[independent_vars + [target_var]].dropna()
    X = regression_data[independent_vars]
    y = regression_data[target_var]
    
    # Split data into 2 sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
    
    # Fit logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    print("Model Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print("Accuracy on Test Data:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model

def perform_extended_logistic_regression(data, extended_independent_vars, target_var='price_change'):
    missing = [var for var in [target_var] + extended_independent_vars if var not in data.columns]
    if missing:
        raise KeyError(f"Missing columns: {', '.join(missing)} in the dataset.")
    
    # Prepare regression data
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)
    regression_data = data[extended_independent_vars + [target_var]].dropna()
    X = regression_data[extended_independent_vars]
    y = regression_data[target_var]

    # Replace infinite and NaN values
    #X.replace([np.inf, -np.inf], np.nan, inplace=True)
    #X.fillna(X.mean(), inplace=True)

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Address class imbalance
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
    
    # Fit logistic regression model
    extended_logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    extended_logistic_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = extended_logistic_model.predict(X_test)

    print("Model Coefficients (Extended):", dict(zip(extended_independent_vars, extended_logistic_model.coef_[0])))
    print("Intercept:", extended_logistic_model.intercept_)
    print("Accuracy on Test Data:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    
    return extended_logistic_model, X_test, y_test

# Perform Random Forest classification
def perform_random_forest(data, independent_vars, target_var='price_change'):
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

    y_pred = rf_model.predict(X_test)
    
    return rf_model

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