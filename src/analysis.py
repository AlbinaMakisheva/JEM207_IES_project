import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import numpy as np
import logging
from src.process_data import apply_lags_and_differencing, create_interaction_terms
from src.visualization import plot_coefficients, align_data, plot_residual_diagnostics

logging.basicConfig(level=logging.INFO)

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
    # Check for missing columns and log them
    missing_vars = [var for var in independent_vars if var not in data.columns]
    if dependent_var not in data.columns:
        missing_vars.append(dependent_var)
    
    if missing_vars:
        logging.error(f"Missing required columns for regression: {', '.join(missing_vars)}")
        raise KeyError(f"Missing required columns for regression: {', '.join(missing_vars)}")
    
    # Perform regression
    regression_data = data[[dependent_var] + independent_vars].dropna()
    X = regression_data[independent_vars]
    y = regression_data[dependent_var]
    
    model = LinearRegression()
    model.fit(X, y)
    
    r2_score = model.score(X, y)
    
    return model, r2_score

# Helper function for residual diagnostics
def plot_residual_diagnostics_for_model(df, independent_vars, dependent_var, model_name="Regression"):
    try:
        st.write(f"Analyzing residuals for {model_name} ...")
        X = df[independent_vars]
        y = df[dependent_var]
        
        # Align data to ensure compatibility
        X_aligned, y_aligned = align_data(X, y)
        
        # Perform the regression
        model, _ = perform_multiple_linear_regression(df, dependent_var, independent_vars)
        
        # Plot residual diagnostics
        plot_residual_diagnostics(model, X_aligned, y_aligned, model_name)
    except Exception as e:
        st.error(f"Error during residual diagnostics for {model_name}: {e}")  
        
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











# Plot lag correlations for all numeric columns
def plot_lag_correlations(data, lag=1):
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"Expected a Pandas DataFrame, but got {type(data)}")

    numeric_data = data.select_dtypes(include=['number'])

    if numeric_data.empty:
        raise ValueError("The DataFrame contains no numeric columns for analysis.")
    
    lagged_data = numeric_data.shift(lag)
    corr_matrix = numeric_data.corrwith(lagged_data, axis=0)
    st.write(f"Autocorrelation at lag {lag} for all variables:")
    st.dataframe(corr_matrix)
    
    for column in numeric_data.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        pd.plotting.lag_plot(numeric_data[column], lag=lag, ax=ax)
        ax.set_title(f'Lag-{lag} Autocorrelation for {column}')
        ax.set_xlabel(f'{column} (t)')
        ax.set_ylabel(f'{column} (t-{lag})')
        st.pyplot(fig)


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


# Calculate feature importance using Random Forest
def calculate_feature_importance(data, target_var):
    X = data.drop(columns=[target_var]).select_dtypes(include='number').fillna(0)
    y = data[target_var]

    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X, y)

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    return feature_importance


# Apply differencing to high-autocorrelation variables
def apply_differencing(data, high_autocorrelation_vars):
    for var in high_autocorrelation_vars:
        data[f'diff_{var}'] = data[var].diff()
    return data


# Perform stationarity tests (ADF test)
def check_stationarity(data):
    numeric_data = data.select_dtypes(include=['number'])
    
    results = []
    for column in numeric_data.columns:
        series = numeric_data[column].dropna()
        
        if series.nunique() <= 1:  
            results.append({
                'Variable': column,
                'ADF Statistic': None,
                'p-value': None,
                'Stationary': 'Constant'
            })
            continue
        
        adf_result = adfuller(series)
        results.append({
            'Variable': column,
            'ADF Statistic': adf_result[0],
            'p-value': adf_result[1],
            'Stationary': 'Yes' if adf_result[1] < 0.05 else 'No'
        })
    
    results_df = pd.DataFrame(results)
    st.write("Stationarity Test Results (ADF Test):")
    st.table(results_df)
    return results_df


# Reduce multicollinearity (Variance Inflation Factor)
def reduce_multicollinearity(data, threshold=10):
    numeric_data = data.select_dtypes(include=['number']).dropna()
    
    vif_data = pd.DataFrame()
    vif_data['Feature'] = numeric_data.columns
    vif_data['VIF'] = [
        variance_inflation_factor(numeric_data.values, i) 
        for i in range(numeric_data.shape[1])
    ]
    
    st.write("Multicollinearity Analysis (VIF):")
    st.table(vif_data)
    
    high_vif_features = vif_data[vif_data['VIF'] > threshold]
    if not high_vif_features.empty:
        st.warning("Features with high VIF (potential multicollinearity):")
        st.table(high_vif_features)
    else:
        st.success("No multicollinearity detected (VIF below threshold).")
    
    return vif_data


# Add lag variables
def add_lagged_features(data, short_lag_vars, long_lag_vars):
    st.header("Adding Lag Variables")
    st.write("""
        Given the high correlation of our variables and their characteristics, we introduce lag variables. 
        Short lags (1-2 weeks) apply to variables like new cases and deaths, whereas long lags (3-6 months) 
        apply to GDP per capita, stringency index, and vaccination signals.
    """)

    short_lags, long_lags = [1], [180]

    for var_group, lags in [(short_lag_vars, short_lags), (long_lag_vars, long_lags)]:
        for var in var_group:
            if var in data.columns:
                diff_var = f"{var}_diff"
                data[diff_var] = data[var].diff()
                for lag in lags:
                    data[f"{diff_var}_lag_{lag}"] = data[diff_var].shift(lag)
            else:
                st.warning(f"Variable {var} not found in data. Skipping.")
    
    data = data.iloc[1:]
    return data


# Perform regression and plot feature importance
def perform_regression_and_plot(data, dependent_var, independent_vars):
    st.header("First Linear Regression")
    regression_model, r2_score = perform_multiple_linear_regression(
        data, dependent_var=dependent_var, independent_vars=independent_vars
    )

    st.subheader("First Regression Results")
    st.markdown(f"**R² Score:** {r2_score:.4f}")

    coefficients_df = pd.DataFrame({
        'Feature': regression_model.feature_names_in_,
        'Coefficient': regression_model.coef_
    }).sort_values(by='Coefficient', ascending=False)
    
    st.table(coefficients_df)

    fig, ax = plt.subplots(figsize=(8, 6))
    coefficients_df.plot.bar(x='Feature', y='Coefficient', legend=False, ax=ax)
    plt.title("Feature Importance (Coefficients)")
    plt.ylabel("Coefficient Value")
    plt.xlabel("Features")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    st.markdown(f"**Intercept:** {regression_model.intercept_:.4f}")


# Display autocorrelation categories
def display_autocorrelation_categories(data):
    st.write("Categorizing variables by autocorrelation...")
    categories = categorize_by_autocorrelation(data, lag=1)

    categorized_vars = {
        "High": categories.get("high", []),
        "Moderate": categories.get("moderate", []),
        "Low": categories.get("low", [])
    }

    st.write("### Autocorrelation Categories")
    for category, vars_list in categorized_vars.items():
        st.write(f"#### {category} Autocorrelation Variables:")
        st.write(vars_list)
    
    return categorized_vars


# Calculate and display feature importance
def calculate_and_display_feature_importance(data, target_var):
    st.write("Calculating feature importance...")
    importance_df = calculate_feature_importance(data, target_var=target_var)
    
    if importance_df is not None and not importance_df.empty:
        st.write("Feature importance calculated:")
        st.table(importance_df)
    else:
        st.warning("Feature importance calculation returned no results.")
    
    return importance_df


# Apply differencing and display
def apply_differencing_and_display(data, high_autocorrelation_vars):
    st.write("Applying differencing to high-autocorrelation variables...")
    diff_data = apply_differencing(data, high_autocorrelation_vars)
    
    st.write("Differenced Data Preview:")
    st.write(diff_data.head())
    
    return diff_data


# Display stationarity test results
def display_stationarity_results(data):
    st.write("Performing stationarity tests...")
    stationarity_results = check_stationarity(data)
    
    st.write("### Stationarity Test Results:")
    st.table(stationarity_results)


# Display VIF results
def display_vif_results(data):
    st.write("Performing multicollinearity analysis...")
    vif_results = reduce_multicollinearity(data)
    
    st.write("### Variance Inflation Factor (VIF):")
    st.table(vif_results)


#Heteroscedasticity Analysis

def test_and_correct_heteroscedasticity(model, X, y, regression_name):
    """
    Test for heteroscedasticity using the Breusch-Pagan test and optionally correct it.
    """ 

    try:
        # Add constant for the test
        X_with_const = sm.add_constant(X)

        # Residuals from the fitted model
        residuals = y - model.predict(X)

        # Perform Breusch-Pagan test
        test_stat, p_value, _, _ = het_breuschpagan(residuals, X_with_const)
        st.write(f"Breusch-Pagan Test for {regression_name}:")
        st.write(f"Test Statistic: {test_stat:.4f}, p-value: {p_value:.4f}")

        if p_value < 0.05:
            st.warning(f"Heteroscedasticity detected for {regression_name}. Correcting using Weighted Least Squares (WLS)...")

            # Correct for heteroscedasticity using WLS
            weights = 1 / (residuals**2 + 1e-8)  # Avoid division by zero
            X_weighted = X_with_const.multiply(weights, axis=0)
            y_weighted = y.multiply(weights)

            # Fit a Weighted Least Squares model
            wls_model = sm.OLS(y_weighted, X_weighted).fit()
            st.write(f"Corrected model for {regression_name}:")
            st.write(wls_model.summary())
            return wls_model
        else:
            st.success(f"No heteroscedasticity detected for {regression_name}. No correction needed.")
            return model

    except Exception as e:
        st.error(f"Error during heteroscedasticity analysis for {regression_name}: {e}")
        return None

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

# Function to perform regression analysis and test for heteroscedasticity
def run_regression_analysis(filtered_data, dependent_var, independent_vars, regression_name):
    try:
        st.subheader(f"{regression_name}")
        # Perform regression
        regression_model, r2_score = perform_multiple_linear_regression(
            filtered_data, dependent_var=dependent_var, independent_vars=independent_vars
        )
        st.write(f"R² Score for {regression_name}: {r2_score:.4f}")

        # Prepare data for heteroscedasticity testing
        X = filtered_data[independent_vars].dropna()
        y = filtered_data[dependent_var].loc[X.index]
        X, y = X.align(y, join="inner", axis=0)

        # Test and correct heteroscedasticity
        test_and_correct_heteroscedasticity(regression_model, X, y, regression_name)
        
    except Exception as e:
        st.error(f"Error during heteroscedasticity analysis for {regression_name}: {e}")
        

# Perform linear regression
def perform_and_display_regression(df, dependent_var, independent_vars):
    regression_model, r2_score = perform_multiple_linear_regression(df, dependent_var, independent_vars)
    st.markdown(f"**R² Score:** {r2_score:.4f}")
    
    coefficients_df = pd.DataFrame({
        'Feature': regression_model.feature_names_in_,
        'Coefficient': regression_model.coef_
    }).sort_values(by='Coefficient', ascending=False)
    
    st.table(coefficients_df)
    plot_coefficients(coefficients_df)
    
    st.markdown(f"**Intercept:** {regression_model.intercept_:.4f}")
    return regression_model


