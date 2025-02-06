import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

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

def prepare_binary_target(df, price_column='close'):
    df['price_change'] = (df[price_column].diff() > 0).astype(int)
    df['target'] = (df[price_column].shift(-1) > df[price_column]).astype(int)
    return df


# Apply differencing and lags to the variables
def process_data_for_regressions(df, short_lags, long_lags):
    reg1_vars_short_lag = ['new_cases_smoothed', 'new_deaths_smoothed', 'reproduction_rate_vaccinations']
    reg1_vars_long_lag = ['vaccination_signal', 'reproduction_rate', 'new_vaccinations_smoothed', 'Dummy_Variable']
    df['reproduction_rate_vaccinations'] = (
        df['reproduction_rate'] * df['vaccination_signal']
    )
    df = apply_lags_and_differencing(df, reg1_vars_short_lag, short_lags)
    df = apply_lags_and_differencing(df, reg1_vars_long_lag, long_lags)
    df = create_interaction_terms(df)
    
    return df


# Apply differencing to high-autocorrelation variables
def apply_differencing(data, high_autocorrelation_vars):
    for var in high_autocorrelation_vars:
        data[f'diff_{var}'] = data[var].diff()
    return data


# Categorize variables by autocorrelation
def categorize_by_autocorrelation(data, lag=1, high_threshold=0.9, moderate_threshold=0.5):
    numeric_data = data.select_dtypes(include=['number'])
    
    autocorrelation_values = {}
    for column in numeric_data.columns:
        autocorrelation = numeric_data[column].autocorr(lag=lag)
        autocorrelation_values[column] = autocorrelation

    high_autocorrelation = [var for var, value in autocorrelation_values.items() if value > high_threshold]
    moderate_autocorrelation = [var for var, value in autocorrelation_values.items() if moderate_threshold < value <= high_threshold]
    low_autocorrelation = [var for var, value in autocorrelation_values.items() if value <= moderate_threshold]

    return {
        'high': high_autocorrelation,
        'moderate': moderate_autocorrelation,
        'low': low_autocorrelation
    }


# Helper function for differencing and lagging variables
def apply_lags_and_differencing(df, variables, lags, differencing=True):
    for var in variables:
        if var in df.columns:
            diff_var = f"{var}_diff" if differencing else var
            if differencing:
                df[diff_var] = df[var].diff()
            for lag in lags:
                df[f"{diff_var}_lag_{lag}"] = df[diff_var].shift(lag)
        else:
            st.write(f"Warning: {var} not found in the data. Skipping.")
    return df

# Helper function for interaction term creation
def create_interaction_terms(df):
    df['new_cases_dummy_interaction'] = df['new_cases_smoothed_diff'] * df['Dummy_Variable']
    df['new_deaths_dummy_interaction'] = df['new_deaths_smoothed_diff'] * df['Dummy_Variable']
    df['deaths_to_cases_ratio'] = np.where(
        df['new_cases_smoothed'] == 0, 0,
        df['new_deaths_smoothed'] / df['new_cases_smoothed']
    )
    
    df['total_vaccination_rate'] = (df['total_vaccinations'] / df['population'])

    df['total_smokers'] = df['female_smokers'].fillna(0) + df['male_smokers'].fillna(0)

    df['female_smokers_rate'] = (df['female_smokers'] / df['total_smokers'])
    return df