import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import streamlit as st
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from src.process_data import apply_lags_and_differencing, create_interaction_terms
from src.visualization import plot_coefficients, align_data, plot_residual_diagnostics

logging.basicConfig(level=logging.INFO)


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


