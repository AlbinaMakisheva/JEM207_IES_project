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
from src.analysis import perform_extended_logistic_regression, perform_and_display_regression, plot_residual_diagnostics_for_model, process_data_for_regressions, run_regression_analysis, filter_data_around_events, test_and_correct_heteroscedasticity, plot_lag_correlations, prepare_binary_target, perform_logistic_regression, perform_random_forest, categorize_by_autocorrelation, calculate_feature_importance, apply_differencing, check_stationarity, reduce_multicollinearity, perform_multiple_linear_regression, add_lagged_features, perform_regression_and_plot, display_autocorrelation_categories, calculate_and_display_feature_importance, apply_differencing_and_display, display_stationarity_results, display_vif_results
from src.visualization import plot_coefficients, plot_residual_diagnostics, plot_covid_cases, plot_stock_with_events, visualize_covid_data, plot_regression_results, plot_roc_curve, display_classification_report, plot_feature_importance, plot_interactive_time_series, plot_scatter_matrix, plot_interactive_heatmap

            
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
    
    # Data loading and filtering
    short_lags = [1]  # Only 1 day for short-term
    long_lags = [180]  # Only 180 days for long-term
        
    # Process data with differencing and lags
    filtered_data = process_data_for_regressions(filtered_data, short_lags, long_lags)
    reg1_independent_vars = (
            [f"{var}_diff_lag_{lag}" for var in ['new_cases_smoothed', 'new_deaths_smoothed'] for lag in short_lags] +
            [f"{var}_diff_lag_{lag}" for var in ['vaccination_signal', 'reproduction_rate', 'new_vaccinations_smoothed', 'Dummy_Variable'] for lag in long_lags] +
            ['new_cases_dummy_interaction', 'new_deaths_dummy_interaction']
    )
    reg2_independent_vars = (
            [f"reproduction_rate_vaccinations_diff_lag_{lag}" for lag in short_lags] +
            [f"{var}_diff_lag_{lag}" for var in ['vaccination_signal', 'Dummy_Variable'] for lag in long_lags] +
            ['deaths_to_cases_ratio', 'new_cases_dummy_interaction', 'new_deaths_dummy_interaction']
    )
    independent_vars = ['new_cases_smoothed', 'Dummy_Variable', 'stringency_index'] 
    independent_vars += ['new_cases_dummy_interaction', 'total_vaccination_rate', 'female_smokers_rate']
    
    try:
        st.header("Add Lag Variables")
        st.write("""
            Given the high correlation in most of our variables and the characteristics of our analysis, we decided to create lag variables. Lag variables account for the time delay between a predictor variable and its effect on the dependent variable (stock returns). We believe that this is highly relevant for variables like vaccinations that take time to impact COVID-19 metrics, which in turn may influence Pfizer's stock performance.
        """)

        # First Linear Regression
        st.header("First Linear Regression")
        regression_model = perform_and_display_regression(filtered_data, dependent_var='daily_return', independent_vars=reg1_independent_vars)
        
        # Second Linear Regression
        st.header("Second Linear Regression")
        regression_model = perform_and_display_regression(filtered_data, dependent_var='daily_return', independent_vars=reg2_independent_vars)
        
        # Third Linear Regression
        st.header("Third Linear Regression")
        regression_model = perform_and_display_regression(filtered_data, dependent_var='new_deaths_smoothed', independent_vars=independent_vars)
        
        # Residual Diagnostics for each regression
        st.header("Residual Diagnostics")
        plot_residual_diagnostics_for_model(filtered_data, reg1_independent_vars, 'daily_return', model_name="First Regression")
        plot_residual_diagnostics_for_model(filtered_data, reg2_independent_vars, 'daily_return', model_name="Second Regression")
        plot_residual_diagnostics_for_model(filtered_data, independent_vars, 'new_deaths_smoothed', model_name="Third Regression")
    except KeyError as e:
        st.error(f"Error during analysis: {e}")
    
    
        # Heteroscedasticity Analysis
    st.header("Heteroscedasticity Analysis")
    st.write("Testing for heteroscedasticity in the regression models and applying corrections if needed...")


    # Running regression analyses
    run_regression_analysis(filtered_data, 'daily_return', reg1_independent_vars, "First Regression")
    run_regression_analysis(filtered_data, 'daily_return', reg2_independent_vars, "Second Regression")
    run_regression_analysis(filtered_data, 'new_deaths_smoothed', independent_vars, "Third Regression")

    # Interpretation of heteroscedasticity results
    st.write("""
    The test statistic and p-values (0.0002, 0.0021, and 0.0000 for the first, second, and third regressions, respectively) indicate heteroscedasticity is present in the residuals of the third regression. 

    Heteroscedasticity implies that the variance of the residuals is not constant, violating a key assumption of OLS. To address this, Weighted Least Squares (WLS) was applied, yielding the following insights:

    - **High R² (uncentered):** Indicates the model explains almost all the variation in the dependent variable. However, caution is needed due to potential multicollinearity or numerical issues.
    - **Significant coefficients:** All predictors have very low p-values (<0.05), suggesting they are statistically significant.
    """)
    
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

        # Extended logistic regression
        # Add new features to the dataset
        merged_data['deaths_to_cases_ratio'] = np.where(
            merged_data['new_cases_smoothed'] == 0, 0,
            merged_data['new_deaths_smoothed'] / merged_data['new_cases_smoothed']
        )
        merged_data['interaction_term'] = merged_data['new_cases_smoothed'] * merged_data['Dummy_Variable']


        # Define the extended independent variables
        extended_independent_vars = independent_vars + ['deaths_to_cases_ratio', 'interaction_term']

        merged_data[extended_independent_vars] = merged_data[extended_independent_vars].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Handle missing or infinite values
        merged_data[extended_independent_vars + ['target']] = merged_data[extended_independent_vars + ['target']].replace(
            [np.inf, -np.inf], np.nan).fillna(0)  

        # Ensure no extreme values exist
        for col in extended_independent_vars:
            merged_data[col] = np.clip(merged_data[col], a_min=-1e6, a_max=1e6)

        # Standardize the independent variables
        scaler = StandardScaler()
        scaled_extended_vars = scaler.fit_transform(merged_data[extended_independent_vars])
        
        # Perform extended logistic regression
        ext_log_model, ext_acc, ext_fpr, ext_tpr, ext_roc_auc, ext_coeffs, _ = perform_extended_logistic_regression(merged_data, extended_independent_vars)
        st.write(f"Extended Logistic Regression Accuracy: {ext_acc}")
        # Classification report for extended logistic regression
        ext_y_true = merged_data['target']
        ext_y_pred = ext_log_model.predict(merged_data[extended_independent_vars])
        display_classification_report(ext_y_true, ext_y_pred, model_name="Extended Logistic Regression")
        plot_roc_curve(ext_fpr, ext_tpr, ext_roc_auc, title="Extended Logistic Regression ROC Curve")

        # Coefficients from the extended logistic regression model
        st.write("Extended Logistic Regression Coefficients:")
        extended_coef_df = pd.DataFrame({
                'Feature': extended_independent_vars,
                'Coefficient': ext_log_model.coef_[0]
            })
        st.table(extended_coef_df)

    except KeyError as e:
        st.error(f"KeyError encountered: {e}")
