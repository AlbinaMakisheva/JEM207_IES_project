import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import streamlit as st

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