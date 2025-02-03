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
from src.analysis import filter_data_around_events, plot_lag_correlations, prepare_binary_target, perform_logistic_regression, perform_random_forest, categorize_by_autocorrelation, calculate_feature_importance, apply_differencing, check_stationarity, reduce_multicollinearity, perform_multiple_linear_regression, add_lagged_features, perform_regression_and_plot, display_autocorrelation_categories, calculate_and_display_feature_importance, apply_differencing_and_display, display_stationarity_results, display_vif_results
from src.visualization import plot_covid_cases, plot_stock_with_events, visualize_covid_data, plot_regression_results, plot_roc_curve, display_classification_report, plot_feature_importance, plot_interactive_time_series, plot_scatter_matrix, plot_interactive_heatmap

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
    return df

# Helper function for plotting regression coefficients
def plot_coefficients(coefficients_df, title="Feature Importance (Coefficients)"):
    fig, ax = plt.subplots(figsize=(8, 6))
    coefficients_df.plot.bar(x='Feature', y='Coefficient', legend=False, ax=ax)
    plt.title(title)
    plt.ylabel("Coefficient Value")
    plt.xlabel("Features")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

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

# Apply differencing and lags to the variables
def process_data_for_regressions(df, short_lags, long_lags):
    reg1_vars_short_lag = ['new_cases_smoothed', 'new_deaths_smoothed']
    reg1_vars_long_lag = ['vaccination_signal', 'reproduction_rate', 'new_vaccinations_smoothed', 'Dummy_Variable']
    
    df = apply_lags_and_differencing(df, reg1_vars_short_lag, short_lags)
    df = apply_lags_and_differencing(df, reg1_vars_long_lag, long_lags)
    
    df = create_interaction_terms(df)
    
    return df

def introduction_tab(merged_data, events, covid_data):
    st.header("Introduction")
    st.write("""
        This project focuses on analyzing the impact of key events on stock prices using event-driven analysis. 
        We examine the relationship between the global COVID-19 pandemic and the stock prices of Pfizer Inc. 
        by investigating the impact of significant events, such as the WHO declaring a pandemic and the approval of the COVID-19 vaccine.
            
        The project involves multiple types of analyses:
        - **Regression Analysis:** We perform multiple linear regression to understand the relationships between stock prices and COVID-related variables.
        - **Event Impact Analysis:** We assess the direct impact of major events on stock returns.
        - **Logistic Regression & Random Forest:** We build models to classify stock movements and identify important features.

        The datasets used in this analysis include global COVID-19 case data and stock price data for Pfizer. These datasets are combined to enable the analysis of event-driven changes in stock behavior.
    """)
        
    # Graph 1: Stock Price with Key Events
    st.write("### Stock Price with Key Events")

    plot_stock_with_events(merged_data, events)

    st.write("""
        - This graph shows **Pfizer's stock price (USD) over time**, with key COVID-19 events marked by vertical dashed lines.
        - **Observations:**
            - **WHO Declares Pandemic (March 2020)**: Pfizer's stock initially experienced **volatility**, showing no immediate uptrend.
            - **First Vaccine Approval (December 2020)**: The stock **rallied significantly**, possibly indicating investor confidence in vaccine-driven revenue.
            - **Vaccination Threshold Reached (July 2021)**: The stock peaked, likely due to strong vaccine sales expectations.
            - **Relaxation of Lockdowns (May 2022)**: Stock **began to decline**, reflecting reduced pandemic-related revenue expectations.
            - **China Easing Zero-COVID Policy (January 2023)**: The downward trend continued as the pandemic's impact on the stock market faded.

        - **Overall Trend:**
            - The stock price was **stable before the pandemic**, **volatile at the start**, **rallied after vaccine approvals**, and **declined post-pandemic** as COVID-related revenues decreased.
    """)

    # Graph 2: Global COVID-19 New Cases
    st.write("### Global COVID-19 New Cases")

    visualize_covid_data(covid_data)

    st.write("""
        - This graph presents **global new COVID-19 cases**, with **red bars representing raw values** and a **blue line for smoothed trends**.
        - **Observations:**
        - **Early waves in 2020**: The number of cases increased significantly after the pandemic declaration.
        - **Major peaks in late 2021 and early 2022**: These align with **Delta and Omicron variant surges**.
        - **Case decline after mid-2022**: Due to **mass vaccinations, natural immunity, and reduced testing**.

        - **Connection to Pfizer Stock Prices:**
        - **Early pandemic surges did not significantly increase Pfizer's stock price**.
        - **The biggest stock price rise happened after vaccine approvals**, not during case surges.
        - **After case peaks and easing of restrictions, Pfizer’s stock declined**, suggesting revenue expectations shifted.
    """)



def analysis_tab(merged_data, events):
    # Filter data around key events
    st.write("Filtering data around key events...")
    window_size = st.slider("Select window size around events (in months)", 1, 12, 3)
    filtered_data = filter_data_around_events(merged_data, events, window_months=window_size)
    
    st.write("""
        The purpose of filtering data around key events is to **analyze patterns or trends before and after these events** to see their effects, such as **stock price movements, changes in COVID-19 cases, etc.** 

        ### Why Filter Data?
        - **It isolates the data** to focus on the periods **directly before and after key events** to study their impact.
        - For example, if analyzing the event **"First Vaccine (2020-12-08)"** with a **1-month window**:
        - The filtered dataset will include data from **2020-11-08 to 2021-01-08**.
        - **Applying this method to all events** creates a dataset segmented into **smaller windows**, allowing a **detailed analysis** of each event's impact.
    """)
    
    try:
        # Perform autocorrelation analysis
        st.write("Performing autocorrelation analysis...")
        plot_lag_correlations(merged_data, lag=1)

        st.write("""
            With this analysis, we aim to investigate the autocorrelation between variables in order to help us building models that predict new trends. 
            It is worth noticing that most variables namely total_cases, new_cases, new_cases_smoothed, total_deaths and new_deaths present high autocorrelation values. This indicate that many of our variables are highly dependent on their previous values at lag 1.
            However, one could also argue that this is a common feature of cumulative variables (which give the cumulative sums over time) and smoothed variables (which are designed to reduce short-term fluctuations).
            Nevertheless, we will focus on variables with much lower correlation, since they are the ones which might add much new information to our regressions.
        """)
    except KeyError as e:
        st.error(f"Error during autocorrelation analysis: {e}")

    try:
        #Categorize variables by autocorrelation
        st.write("Categorizing variables by autocorrelation...")

        autocorrelation_categories = categorize_by_autocorrelation(merged_data, lag=1)

        # Access the categorized variables
        high_autocorrelation_vars = autocorrelation_categories['high']
        moderate_autocorrelation_vars = autocorrelation_categories['moderate']
        low_autocorrelation_vars = autocorrelation_categories['low']

        # Display results in Streamlit
        st.write("### Autocorrelation Categories")
        st.write("#### High Autocorrelation Variables:")
        st.write(high_autocorrelation_vars)
        st.write("#### Moderate Autocorrelation Variables:")
        st.write(moderate_autocorrelation_vars)
        st.write("#### Low Autocorrelation Variables:")
        st.write(low_autocorrelation_vars)

        st.write("Select relevant variables based on importance...")

        #Calculate feature importance using Random Forest
        st.write("Calculating feature importance...")
        feature_importance = calculate_feature_importance(merged_data, target_var='daily_return') 
        st.write("Feature importance calculated:")
        st.table(feature_importance)

        top_variables = feature_importance['Feature'].head(10).tolist() 
        critical_high_autocorrelation_vars = [var for var in high_autocorrelation_vars if var in top_variables]

        st.write("Critical High Autocorrelation Variables:", critical_high_autocorrelation_vars)
            
        # Apply differencing to critical variables
        st.write("Applying differencing to critical variables...")
        merged_data = apply_differencing(merged_data, critical_high_autocorrelation_vars)

        st.write("Differencing applied. New columns added:")
        st.write([f'diff_{var}' for var in critical_high_autocorrelation_vars])

    except KeyError as e:
        st.error(f"Error during analysis: {e}")


    try:
        #Combine all variables for testing
        test_variables = (
            [f'diff_{var}' for var in critical_high_autocorrelation_vars] +
            moderate_autocorrelation_vars +
            low_autocorrelation_vars
        )

        #Check Stationarity on selected variables: Augmented Dickey-Fuller Test
        # st.write("Perfoming Stationarity...")
        # stationarity_results = check_stationarity(merged_data[test_variables])
        # st.table(stationarity_results)

        # Reduce multicollinearity
        st.write("Analyzing multicollinearity (VIF)...")
        vif_results = reduce_multicollinearity(merged_data[test_variables], threshold=10)
        st.table(vif_results)
        
    except KeyError as e:
        st.error(f"Error during stationarity and multicollinearity analysis: {e}")

    try:
         # Data loading and filtering
        st.header("Add Lag Variables")
        st.write("""
            Given the high correlation in most of our variables and the characteristics of our analysis, we decided to create lag variables. Lag variables account for the time delay between a predictor variable and its effect on the dependent variable (stock returns). We believe that this is highly relevant for variables like vaccinations that take time to impact COVID-19 metrics, which in turn may influence Pfizer's stock performance.
        """)

        short_lags = [1]  # Only 1 day for short-term
        long_lags = [180]  # Only 180 days for long-term
        
        # Process data with differencing and lags
        filtered_data = process_data_for_regressions(filtered_data, short_lags, long_lags)
        
        # First Linear Regression
        st.header("First Linear Regression")
        reg1_independent_vars = (
            [f"{var}_diff_lag_{lag}" for var in ['new_cases_smoothed', 'new_deaths_smoothed'] for lag in short_lags] +
            [f"{var}_diff_lag_{lag}" for var in ['vaccination_signal', 'reproduction_rate', 'new_vaccinations_smoothed', 'Dummy_Variable'] for lag in long_lags] +
            ['new_cases_dummy_interaction', 'new_deaths_dummy_interaction']
        )
        regression_model = perform_and_display_regression(filtered_data, dependent_var='daily_return', independent_vars=reg1_independent_vars)
        
        # Second Linear Regression
        st.header("Second Linear Regression")
        reg2_independent_vars = (
            [f"reproduction_rate_vaccinations_lag_{lag}" for lag in short_lags] +
            [f"{var}_lag_{lag}" for var in ['vaccination_signal', 'Dummy_Variable'] for lag in long_lags] +
            ['deaths_to_cases_ratio', 'new_cases_dummy_interaction', 'new_deaths_dummy_interaction']
        )
        regression_model = perform_and_display_regression(filtered_data, dependent_var='daily_return', independent_vars=reg2_independent_vars)
        
        # Third Linear Regression
        st.header("Third Linear Regression")
        independent_vars = ['new_cases_smoothed', 'Dummy_Variable', 'stringency_index'] 
        independent_vars += ['new_cases_dummy_interaction', 'total_vaccination_rate', 'female_smokers_rate']
        regression_model = perform_and_display_regression(filtered_data, dependent_var='new_deaths_smoothed', independent_vars=independent_vars)
        
        # Residual Diagnostics for each regression
        st.header("Residual Diagnostics")
        plot_residual_diagnostics_for_model(filtered_data, reg1_independent_vars, 'daily_return', model_name="First Regression")
        plot_residual_diagnostics_for_model(filtered_data, reg2_independent_vars, 'daily_return', model_name="Second Regression")
        plot_residual_diagnostics_for_model(filtered_data, independent_vars, 'new_deaths_smoothed', model_name="Third Regression")
    except KeyError as e:
        st.error(f"Error during analysis: {e}")
        
    try:
        merged_data = prepare_binary_target(filtered_data, price_column='close') 
        independent_vars = ['new_vaccinations_smoothed', 'new_deaths_smoothed', 'new_cases_smoothed', 'Dummy_Variable']

        merged_data = merged_data.dropna(subset=independent_vars + ['target'])
        # imputer = SimpleImputer(strategy='mean')
        # merged_data[independent_vars] = imputer.fit_transform(merged_data[independent_vars])

        # Perform logistic regression
        log_model, log_acc, log_fpr, log_tpr, log_roc_auc = perform_logistic_regression(
            merged_data, independent_vars
        )
        st.write(f"Logistic Regression Accuracy: {log_acc}")
        # Display the classification report for logistic regression
        log_y_true = merged_data['target']
        log_y_pred = log_model.predict(merged_data[independent_vars])
        display_classification_report(log_y_true, log_y_pred, model_name="Logistic Regression")
        plot_roc_curve(log_fpr, log_tpr, log_roc_auc, title="Logistic Regression ROC Curve")

        # Perform random forest
        rf_model, rf_acc, rf_fpr, rf_tpr, rf_roc_auc = perform_random_forest(merged_data, independent_vars)
        st.write(f"Random Forest Accuracy: {rf_acc}")
        # Display the classification report for random forest
        rf_y_true = merged_data['target']
        rf_y_pred = rf_model.predict(merged_data[independent_vars])
        display_classification_report(rf_y_true, rf_y_pred, model_name="Random Forest")
        plot_roc_curve(rf_fpr, rf_tpr, rf_roc_auc, title="Random Forest ROC Curve")
    except KeyError as e:
        st.error(f"Error during analysis: {e}")

        
  