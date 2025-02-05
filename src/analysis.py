import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from .data_merging import merge_data
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan


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

#Categorize Variables by Autocorrelation

def categorize_by_autocorrelation(data, lag=1, high_threshold=0.9, moderate_threshold=0.5):
    """
    Categorize variables into high, moderate, and low autocorrelation groups based on their autocorrelation values.

    """
    # Select numeric columns only
    numeric_data = data.select_dtypes(include=['number'])
    
    # Compute autocorrelations
    autocorrelation_values = {}
    for column in numeric_data.columns:
        autocorrelation = numeric_data[column].autocorr(lag=lag)
        autocorrelation_values[column] = autocorrelation

    # Categorize variables
    high_autocorrelation = [var for var, value in autocorrelation_values.items() if value > high_threshold]
    moderate_autocorrelation = [var for var, value in autocorrelation_values.items() if moderate_threshold < value <= high_threshold]
    low_autocorrelation = [var for var, value in autocorrelation_values.items() if value <= moderate_threshold]

    # Return categorized variables
    return {
        'high': high_autocorrelation,
        'moderate': moderate_autocorrelation,
        'low': low_autocorrelation
    }

#Calculate feature importance using Random Forest

def calculate_feature_importance(data, target_var):
    # Split dataset into features (X) and target (y)
    X = data.drop(columns=[target_var]).select_dtypes(include='number').fillna(0)  # Fill NaN values
    y = data[target_var]

    rf = RandomForestRegressor(
        n_estimators=50,   # Reduce number of trees 
        max_depth=10,      # Limit depth of each tree
        random_state=42
    )
    
    rf.fit(X, y)

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    return feature_importance


#Apply Differencing to High Autocorrelation Variables

def apply_differencing(data, high_autocorrelation_vars):
    """
    Apply differencing to high-autocorrelation variables to make them stationary.

    """
    for var in high_autocorrelation_vars:
        # Create a new column for the differenced variable
        data[f'diff_{var}'] = data[var].diff()
    
    return data

#Perfoming Stationarity: Augmented Dickey-Fuller Test

def check_stationarity(data):
    # Filter numeric columns
    numeric_data = data.select_dtypes(include=['number'])
    
    results = []
    for column in numeric_data.columns:
        series = numeric_data[column].dropna()
        
        # Skip constant columns
        if series.nunique() <= 1:  
            results.append({
                'Variable': column,
                'ADF Statistic': None,
                'p-value': None,
                'Stationary': 'Constant'
            })
            continue
        
        # Perform ADF test
        adf_result = adfuller(series)
        results.append({
            'Variable': column,
            'ADF Statistic': adf_result[0],
            'p-value': adf_result[1],
            'Stationary': 'Yes' if adf_result[1] < 0.05 else 'No'
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

#Residual Diagnostics

def plot_residual_diagnostics(model, X, y, regression_name):
    try:
        # Ensure features match those used during fit
        if list(model.feature_names_in_) != list(X.columns):
            missing_features = set(model.feature_names_in_) - set(X.columns)
            unexpected_features = set(X.columns) - set(model.feature_names_in_)
            raise ValueError(
                f"Feature mismatch for {regression_name}:\n"
                f"Missing features: {missing_features}\n"
                f"Unexpected features: {unexpected_features}"
            )

        # Predict and calculate residuals
        predictions = model.predict(X)
        residuals = y - predictions

        # Plot Residual Diagnostics
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Residuals vs Fitted
        sns.scatterplot(x=predictions, y=residuals, ax=axs[0], color="blue", alpha=0.6)
        axs[0].axhline(0, color="red", linestyle="--", linewidth=1)
        axs[0].set_title("Residuals vs Fitted")
        axs[0].set_xlabel("Fitted Values")
        axs[0].set_ylabel("Residuals")

        # Histogram of Residuals
        sns.histplot(residuals, kde=True, bins=20, ax=axs[1], color="blue", alpha=0.6)
        axs[1].axhline(0, color="red", linestyle="--", linewidth=1)
        axs[1].set_title("Distribution of Residuals")

        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during residual diagnostics for {regression_name}: {e}")

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
    

# Function to display the results in Streamlit
def display_analysis_results(regression_model, r2_score, event_impact, rf_model):
    st.subheader("Regression Results")
    st.write(f"R²: {r2_score:.4f}")
    st.write("Coefficients:")
    st.write(dict(zip(regression_model.feature_names_in_, regression_model.coef_)))
    st.write(f"Intercept: {regression_model.intercept_:.4f}")
    
    st.subheader("Event Impact Analysis")
    st.write(event_impact)
    