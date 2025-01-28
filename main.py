import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from src.data_cleaning import clean_data
from src.data_merging import merge_data
import matplotlib.pyplot as plt 
from src.analysis import filter_data_around_events, plot_lag_correlations, categorize_by_autocorrelation, calculate_feature_importance, apply_differencing, check_stationarity, reduce_multicollinearity, perform_multiple_linear_regression, analyze_event_impact, prepare_binary_target, perform_logistic_regression, perform_extended_logistic_regression, perform_random_forest, plot_residual_diagnostics, test_and_correct_heteroscedasticity
from src.visualization import plot_covid_cases, plot_stock_with_events, visualize_covid_data, plot_regression_results, plot_interactive_time_series, plot_scatter_matrix, plot_interactive_heatmap
from src.data_fetching import fetch_covid_data, fetch_stock_data
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler


# File paths
COVID_FILE = 'data/raw/covid_data.csv'
STOCK_FILE = 'data/raw/pfizer_stock.csv'


def main():
    st.title("Event-Driven Analysis")

    # Key events
    events = {
        "WHO Declares Pandemic": "2020-03-11",
        "First Vaccine": "2020-12-08",
        "Vaccination Threshold Reached (85%)": "2021-07-30",
        "Relaxation of Lockdowns": "2022-05-01",  
        "Omicron-Specific Vaccine Approval": "2022-09-01", 
        "China Eases Zero-COVID Policy": "2023-01-01", 
    }

    # Fetch data 
    if not os.path.exists(COVID_FILE):
        st.write("Fetching COVID data...")
        fetch_covid_data(COVID_FILE)
        
    if not os.path.exists(STOCK_FILE):
        st.write("Fetching stock data...")
        fetch_stock_data(STOCK_FILE)

    # Clean data
    covid_data = clean_data(COVID_FILE)
    stock_data = clean_data(STOCK_FILE, is_stock=True) 

    merged_data = merge_data(covid_data, stock_data, events) 

    # Apply log transformation to selected variables
    variables_to_log_transform = ['new_deaths_smoothed', 'new_cases_smoothed']
    non_negative_data = merged_data[variables_to_log_transform]
    merged_data[variables_to_log_transform] = np.log1p(non_negative_data)

    # Create tabs
    tab = st.sidebar.radio("Select a Tab", ("Introduction", "Analysis", "COVID-19 Map"))

    if tab == "Introduction":
        st.header("Introduction to the Project")
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

    if tab == "Analysis":
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
            st.write("Perfoming Stationarity...")
            stationarity_results = check_stationarity(merged_data[test_variables])
            st.table(stationarity_results)

            # Reduce multicollinearity
            st.write("Analyzing multicollinearity (VIF)...")
            vif_results = reduce_multicollinearity(merged_data[test_variables], threshold=10)
            st.table(vif_results)
        
        except KeyError as e:
            st.error(f"Error during stationarity and multicollinearity analysis: {e}")


        try:
            #Adding lag variables

            st.header("Add Lag Variables")
            st.write("""
                Given the high correlation in most of our variables and the caracteristics of our analysis, we decided to create lag variables. Lag variables account for the time delay between a predictor variable and its effect on the dependent variable (stock returns). We believe that this is highly relevant for variables like vaccinations that take time to impact COVID-19 metrics which in turn may influence Pfizer's stock performance. For highly autocorrelated variables, their lagged values might add predictive power.

                Given this, we also decided to separate the variables to account for the time it would take to impact the outcome. Long lags (3-6 months) might be helpful for variables like gdp_per_capita, stringency_index, vaccination rates, or vaccination signals, as their effects are often gradual. Variables like new_cases and new_deaths typically have shorter lags (e.g., 1–2 weeks) since the market reacts quickly to changes in pandemic severity.
            """)

            # First Linear Regression
            st.header("First Linear Regression")

            # Apply differencing and lagging to selected variables
            reg1_vars_short_lag = ['new_cases_smoothed', 'new_deaths_smoothed']
            reg1_vars_long_lag = ['vaccination_signal', 'reproduction_rate', 'new_vaccinations_smoothed', 'Dummy_Variable']

            st.write("Applying differencing and lagging to selected variables...")
            short_lags = [1]  # Only 1 day for short-term
            long_lags = [180]  # Only 180 days for long-term

            # Apply differencing and short lags to short-term variables
            for var in reg1_vars_short_lag:
                if var in filtered_data.columns:
                    diff_var = f"{var}_diff"
                    filtered_data[diff_var] = filtered_data[var].diff()
                    for lag in short_lags:
                        filtered_data[f"{diff_var}_lag_{lag}"] = filtered_data[diff_var].shift(lag)
                else:
                    st.write(f"Warning: {var} not found in the data. Skipping.")

            # Apply differencing and long lags to long-term variables
            for var in reg1_vars_long_lag:
                if var in filtered_data.columns:
                    diff_var = f"{var}_diff"
                    filtered_data[diff_var] = filtered_data[var].diff()
                    for lag in long_lags:
                        filtered_data[f"{diff_var}_lag_{lag}"] = filtered_data[diff_var].shift(lag)
                else:
                    st.write(f"Warning: {var} not found in the data. Skipping.")

            filtered_data = filtered_data.iloc[1:]  # Removes the first row affected by differencing

            # Create interaction terms
            filtered_data['new_cases_dummy_interaction'] = (
                filtered_data['new_cases_smoothed_diff'] * filtered_data['Dummy_Variable']
            )

            filtered_data['new_deaths_dummy_interaction'] = (
                filtered_data['new_deaths_smoothed_diff'] * filtered_data['Dummy_Variable']
            )

            reg1_independent_vars = (
                [f"{var}_diff_lag_{lag}" for var in reg1_vars_short_lag for lag in short_lags] +
                [f"{var}_diff_lag_{lag}" for var in reg1_vars_long_lag for lag in long_lags] +
                ['new_cases_dummy_interaction', 'new_deaths_dummy_interaction']
            )

            # Perform regression
            regression_model, r2_score = perform_multiple_linear_regression(
                filtered_data,
                dependent_var='daily_return',
                independent_vars=reg1_independent_vars
            )

            st.subheader("First Regression Results")
            st.markdown(f"**R² Score:** {r2_score:.4f}")
            coefficients_df = pd.DataFrame({
                'Feature': regression_model.feature_names_in_,
                'Coefficient': regression_model.coef_
            }).sort_values(by='Coefficient', ascending=False)
            st.table(coefficients_df)

            # Plot Coefficients
            st.markdown("**Feature Importance (Coefficients):**")
            fig, ax = plt.subplots(figsize=(8, 6))
            coefficients_df.plot.bar(
                x='Feature', y='Coefficient', legend=False, ax=ax
            )
            plt.title("Feature Importance (Coefficients)")
            plt.ylabel("Coefficient Value")
            plt.xlabel("Features")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

            # Intercept
            st.markdown(f"**Intercept:** {regression_model.intercept_:.4f}")

            # Second Linear Regression
            st.header("Second Linear Regression")

            # Create additional interaction terms
            filtered_data['new_cases_dummy_interaction'] = (
                filtered_data['new_cases_smoothed'] * filtered_data['Dummy_Variable']
            )

            filtered_data['reproduction_rate_vaccinations'] = (
                filtered_data['reproduction_rate'] * filtered_data['vaccination_signal']
            )

            filtered_data['deaths_to_cases_ratio'] = np.where(
                filtered_data['new_cases_smoothed'] == 0, 0,
                filtered_data['new_deaths_smoothed'] / filtered_data['new_cases_smoothed']
            )

            filtered_data['new_deaths_dummy_interaction'] = (
                filtered_data['new_deaths_smoothed'] * filtered_data['Dummy_Variable']
            )


            reg2_vars_short_lag = ['reproduction_rate_vaccinations']
            reg2_vars_long_lag = ['vaccination_signal', 'Dummy_Variable']

            # Apply differencing and short lags to short-term variables
            for var in reg2_vars_short_lag:
                if var in filtered_data.columns:
                    for lag in short_lags:
                        filtered_data[f"{var}_lag_{lag}"] = filtered_data[var].shift(lag)
                else:
                    st.write(f"Warning: {var} not found in the data. Skipping.")

            # Apply differencing and long lags to long-term variables
            for var in reg2_vars_long_lag:
                if var in filtered_data.columns:
                    for lag in long_lags:
                        filtered_data[f"{var}_lag_{lag}"] = filtered_data[var].shift(lag)
                else:
                    st.write(f"Warning: {var} not found in the data. Skipping.")

            reg2_independent_vars = (
                [f"{var}_lag_{lag}" for var in reg2_vars_short_lag for lag in short_lags] +
                [f"{var}_lag_{lag}" for var in reg2_vars_long_lag for lag in long_lags] +
                ['deaths_to_cases_ratio', 'new_cases_dummy_interaction', 'new_deaths_dummy_interaction']
            )

            regression_model, r2_score = perform_multiple_linear_regression(
                filtered_data,
                dependent_var='daily_return',
                independent_vars=reg2_independent_vars
            )

            st.subheader("Second Regression Results")
            st.markdown(f"**R² Score:** {r2_score:.4f}")
            coefficients_df = pd.DataFrame({
                'Feature': regression_model.feature_names_in_,
                'Coefficient': regression_model.coef_
            }).sort_values(by='Coefficient', ascending=False)
            st.table(coefficients_df)

            # Plot Coefficients
            st.markdown("**Feature Importance (Coefficients):**")
            fig, ax = plt.subplots(figsize=(8, 6))
            coefficients_df.plot.bar(
                x='Feature', y='Coefficient', legend=False, ax=ax
            )
            plt.title("Feature Importance (Coefficients)")
            plt.ylabel("Coefficient Value")
            plt.xlabel("Features")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

            # Intercept
            st.markdown(f"**Intercept:** {regression_model.intercept_:.4f}")


            # Third Linear Regression
            st.header("Third Linear Regression")

            independent_vars = ['new_cases_smoothed', 'Dummy_Variable', 'stringency_index'] 


            # Create new interaction terms: 
            filtered_data['new_cases_dummy_interaction'] = (
                filtered_data['new_cases_smoothed'] * filtered_data['Dummy_Variable']
            )

            filtered_data['total_vaccination_rate'] = (filtered_data['total_vaccinations'] / filtered_data['population'])

            filtered_data['total_smokers'] = filtered_data['female_smokers'].fillna(0) + merged_data['male_smokers'].fillna(0)

            filtered_data['female_smokers_rate'] = (filtered_data['female_smokers'] / filtered_data['total_smokers'])


            # Prepare the full list of independent variables for regression
            independent_vars = independent_vars + ['new_cases_dummy_interaction'] + ['total_vaccination_rate'] + ['female_smokers_rate'] 
            # Perform multiple linear regression
            st.write("Performing Regression Analysis...")
            regression_model, r2_score = perform_multiple_linear_regression(
                filtered_data,
                dependent_var='new_deaths_smoothed',
                independent_vars=independent_vars
            )

            # Display regression results
            st.subheader("Regression Results")

            # R² Score
            st.markdown(f"**R² Score:** {r2_score:.4f}")

            # Coefficients Table
            coefficients_df = pd.DataFrame({
                'Feature': regression_model.feature_names_in_,
                'Coefficient': regression_model.coef_
            }).sort_values(by='Coefficient', ascending=False)
            st.markdown("**Coefficients:**")
            st.table(coefficients_df)

            # Plot Coefficients
            st.markdown("**Feature Importance (Coefficients):**")
            fig, ax = plt.subplots(figsize=(8, 6))
            coefficients_df.plot.bar(
                x='Feature', y='Coefficient', legend=False, ax=ax
            )
            plt.title("Feature Importance (Coefficients)")
            plt.ylabel("Coefficient Value")
            plt.xlabel("Features")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

            # Intercept
            st.markdown(f"**Intercept:** {regression_model.intercept_:.4f}")

            # Residual Diagnostics for Linear Regressions
            st.header("Residual Diagnostics")
            st.write("The purpose of this analysis is to visualize residuals to check the goodness-of-fit of regression models")

            # Helper function to ensure X and y have aligned rows
            def align_data(X, y):
                """Align X and y by their indices to ensure compatibility."""
                aligned_data = pd.concat([X, y], axis=1).dropna()
                return aligned_data[X.columns], aligned_data[y.name]

            # Residuals for the first regression
            try:
                st.write("Analyzing residuals for the first regression ...")
                # Define the independent and dependent variables
                X_reg1 = filtered_data[reg1_independent_vars]
                y_reg1 = filtered_data['daily_return']
                # Align data
                X_reg1_aligned, y_reg1_aligned = align_data(X_reg1, y_reg1)
                # Perform the regression
                model_reg1, _ = perform_multiple_linear_regression(filtered_data, 'daily_return', reg1_independent_vars)
                # Plot residual diagnostics
                plot_residual_diagnostics(model_reg1, X_reg1_aligned, y_reg1_aligned, "First Regression")
            except Exception as e:
                st.error(f"Error during residual diagnostics for First Regression: {e}")

            # Residuals for the second regression
            try:
                st.write("Analyzing residuals for the second regression ...")
                # Define the independent and dependent variables
                X_reg2 = filtered_data[reg2_independent_vars]
                y_reg2 = filtered_data['daily_return']
                # Align data
                X_reg2_aligned, y_reg2_aligned = align_data(X_reg2, y_reg2)
                # Perform the regression
                model_reg2, _ = perform_multiple_linear_regression(filtered_data, 'daily_return', reg2_independent_vars)
                # Plot residual diagnostics
                plot_residual_diagnostics(model_reg2, X_reg2_aligned, y_reg2_aligned, "Second Regression")
            except Exception as e:
                st.error(f"Error during residual diagnostics for Second Regression: {e}")

            # Residuals for the third regression
            try:
                st.write("Analyzing residuals for the third regression ...")
                # Define the independent and dependent variables
                X_reg3 = filtered_data[independent_vars]
                y_reg3 = filtered_data['new_deaths_smoothed']
                # Align data
                X_reg3_aligned, y_reg3_aligned = align_data(X_reg3, y_reg3)
                # Perform the regression
                model_reg3, _ = perform_multiple_linear_regression(filtered_data, 'new_deaths_smoothed', independent_vars)
                # Plot residual diagnostics
                plot_residual_diagnostics(model_reg3, X_reg3_aligned, y_reg3_aligned, "Third Regression")
            except Exception as e:
                st.error(f"Error during residual diagnostics for Third Regression: {e}")


            #Observations from the Residual Diagnostics
            st.write("Observations from the Residual Diagnostics...")

            st.write("""
                    In the first regression,the residuals seem relatively centered around 0, but there appears to be a pattern at lower fitted values.
                    For the second regression, there is a noticeable clustering of residuals at certain ranges of fitted values. This indicates potential issues with heteroscedasticity (non-constant variance) or omitted variable bias.
                    For the third regression, the residuals display a strong pattern,  which could be understood as a sign of heteroscedasticity.The variance of residuals increases with higher fitted values, indicating that the model struggles to capture variability at those levels.
                    In summary, through graphical visualization, the residuals appear to increase, indicating that they are not constant over time.
                    
                    Regarding the Distribution of Residuals, the residuals of all three regressions are approximately normal but have visible tails or are slightly skewed. This indicates a reasonable fit but with room for improvement in capturing the underlying patterns.
                    Given this, and to improve our regressions, we should apply the necessary corrections in order to restore the classical assumption of constant variance. It is also important to highlight that these procedures, while not fully resolving the problem of heteroscedasticity, restore the validity of statistical inference in large samples.
                    """)
           
            #Heteroscedasticity analysis

            st.header("Heteroscedasticity Analysis")
            st.write("Testing for heteroscedasticity in the regression models and applying corrections if needed...")

            try:
                # First Regression
                st.subheader("First Regression")
                # Perform regression to retrieve the model
                regression_model_1, r2_score_1 = perform_multiple_linear_regression(
                    filtered_data,
                    dependent_var='daily_return',
                    independent_vars=reg1_independent_vars
                )
                st.write(f"R² Score for First Regression: {r2_score_1:.4f}")

                # Prepare data for heteroscedasticity testing
                X_reg1 = filtered_data[reg1_independent_vars].dropna()
                y_reg1 = filtered_data['daily_return'].loc[X_reg1.index]
                X_reg1, y_reg1 = X_reg1.align(y_reg1, join="inner", axis=0)

                # Test and correct for heteroscedasticity
                test_and_correct_heteroscedasticity(regression_model_1, X_reg1, y_reg1, "First Regression")
            except Exception as e:
                st.error(f"Error during heteroscedasticity analysis for First Regression: {e}")

            try:
                # Second Regression
                st.subheader("Second Regression")
                # Perform regression to retrieve the model
                regression_model_2, r2_score_2 = perform_multiple_linear_regression(
                    filtered_data,
                    dependent_var='daily_return',
                    independent_vars=reg2_independent_vars
                )
                st.write(f"R² Score for Second Regression: {r2_score_2:.4f}")

                # Prepare data for heteroscedasticity testing
                X_reg2 = filtered_data[reg2_independent_vars].dropna()
                y_reg2 = filtered_data['daily_return'].loc[X_reg2.index]
                X_reg2, y_reg2 = X_reg2.align(y_reg2, join="inner", axis=0)

                # Test and correct for heteroscedasticity
                test_and_correct_heteroscedasticity(regression_model_2, X_reg2, y_reg2, "Second Regression")
            except Exception as e:
                st.error(f"Error during heteroscedasticity analysis for Second Regression: {e}")

            try:
                
                # Handle missing values for the third regression
                st.subheader("Third Regression")
                X_reg3 = filtered_data[independent_vars]
                y_reg3 = filtered_data['new_deaths_smoothed']
                X_reg3 = X_reg3.dropna()
                y_reg3 = y_reg3[X_reg3.index]
                corrected_model_3 = test_and_correct_heteroscedasticity(regression_model, X_reg3, y_reg3, "Third Regression")

                st.write("""
                The test statistic and a p-value of 0.0002, 0.0021 and 0.0000 (for the first, second and third regression respectively), indicate heteroscedasticity is present in the residuals of the third regression. Heteroscedasticity implies that the variance of the residuals is not constant, violating a key assumption of ordinary least squares (OLS). Weighted Least Squares (WLS) was applied to address heteroscedasticity, and a corrected model was obtained.
               
                The corrected model's results show:
                         
                >High R² (uncentered): Indicates the model explains almost all the variation in the dependent variable. However, this should be interpreted cautiously due to potential multicollinearity or numerical issues.
                         
                >Significant coefficients: All predictors have very low p-values (<0.05), suggesting they are statistically significant.
                         """)

            except Exception as e:
                st.error(f"Error during heteroscedasticity analysis for Second Regression {e}")


        except Exception as e:
            st.error(f"An error occurred: {e}")

        try: 

            # Prepare binary target for logistic regression
            merged_data = prepare_binary_target(filtered_data, price_column='close') 
            independent_vars = ['new_vaccinations_smoothed', 'new_deaths_smoothed', 'new_cases_smoothed', 'Dummy_Variable']

            # Perform logistic regression with additional features
            st.write("Performing Logistic Regression with Extended Variables...")
            
            # Add new features to the dataset
            merged_data['deaths_to_cases_ratio'] = np.where(
                merged_data['new_cases_smoothed'] == 0, 0,
                merged_data['new_deaths_smoothed'] / merged_data['new_cases_smoothed']
            )
            merged_data['interaction_term'] = merged_data['new_cases_smoothed'] * merged_data['Dummy_Variable']


            # Define the extended independent variables
            extended_independent_vars = independent_vars + ['deaths_to_cases_ratio', 'interaction_term']

            # Handle missing or infinite values in the extended variables
            merged_data[extended_independent_vars + ['target']] = merged_data[extended_independent_vars + ['target']].replace(
                [np.inf, -np.inf], np.nan).fillna(0)  

            # Ensure no extreme values exist
            for col in extended_independent_vars:
                merged_data[col] = np.clip(merged_data[col], a_min=-1e6, a_max=1e6)

            # Standardize the independent variables
            scaler = StandardScaler()
            scaled_extended_vars = scaler.fit_transform(merged_data[extended_independent_vars])

            # Perform extended logistic regression
        
            extended_logistic_model, X_test, y_test = perform_extended_logistic_regression(merged_data, extended_independent_vars, target_var='target')

            # Display results extended logistic regression
            st.subheader("Extended Logistic Regression Results")
            st.markdown("**Model Coefficients:**")
            extended_coef_df = pd.DataFrame({
                'Feature': extended_independent_vars,
                'Coefficient': extended_logistic_model.coef_[0]
            })
            st.table(extended_coef_df)

            st.markdown(f"**Intercept:** {extended_logistic_model.intercept_}")

            extended_y_pred = extended_logistic_model.predict(scaled_extended_vars)
            extended_accuracy = accuracy_score(merged_data['target'], extended_y_pred)
            st.markdown(f"**Accuracy on Test Data:** {extended_accuracy:.4f}")

            # Classification report for extended logistic regression
            from sklearn.metrics import classification_report
            extended_report = classification_report(y_test, extended_logistic_model.predict(X_test), output_dict=True, zero_division=0)
            extended_report_df = pd.DataFrame(extended_report).transpose()
            st.markdown("**Classification Report (Extended):**")
            st.table(extended_report_df)

            st.markdown("### Model Performance Plots")

            # Calculate ROC Curve
            from sklearn.metrics import roc_curve, auc

            if extended_logistic_model.classes_.shape[0] > 1:
                y_proba = extended_logistic_model.predict_proba(X_test)[:, 1]
            else:
                st.error("ROC Curve cannot be computed: Model outputs only one class.")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            st.write(f"**ROC AUC:** {roc_auc:.4f}")

            # Plot ROC Curve
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Extended Logistic Regression ROC Curve')
            ax.legend(loc='lower right')
            st.pyplot(fig)

            # Perform Random Forest classification
            st.write("Performing Random Forest Classification...")
            rf_model_2 = perform_random_forest(merged_data, extended_independent_vars)

            # Display the feature importance
            feature_importance = pd.DataFrame({
                'Feature': extended_independent_vars,  
                'Importance': rf_model_2.feature_importances_
            })
            st.subheader("Random Forest Feature Importance")
            st.table(feature_importance)

            # Plot the feature importance
            feature_importance.plot(kind='barh', x='Feature', y='Importance', legend=False)
            plt.title('Random Forest Feature Importance')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.show()

            feature_importance.plot(kind='barh', x='Feature', y='Importance')
            st.pyplot(plt)

            # Perform logistic regression with additional features with diff-lag
            st.write("Performing Logistic Regression with Extended Variables applying diff and lag...")

            # Define short lag and long lag variables
            vars_short_lag = ['new_cases_smoothed', 'new_deaths_smoothed']
            vars_long_lag = ['new_vaccinations_smoothed', 'Dummy_Variable']

            # Define lag periods
            short_lags = [1]  # 1-week lag
            long_lags = [180]  # 6-month lag

            # Apply short lags to short-term variables
            for var in vars_short_lag:
                if var in merged_data.columns:
                    diff_var = f"{var}_diff"
                    merged_data[diff_var] = merged_data[var].diff()
                    for lag in short_lags:
                        merged_data[f"{diff_var}_lag_{lag}"] = merged_data[diff_var].shift(lag)
                else:
                    st.write(f"Warning: {var} not found in the data. Skipping.")

            # Apply long lags to long-term variables
            for var in vars_long_lag:
                if var in merged_data.columns:
                    diff_var = f"{var}_diff"
                    merged_data[diff_var] = merged_data[var].diff()
                    for lag in long_lags:
                        merged_data[f"{diff_var}_lag_{lag}"] = merged_data[diff_var].shift(lag)
                else:
                    st.write(f"Warning: {var} not found in the data. Skipping.")

            # Define independent variables using only differenced and lagged versions
            independent_vars = (
                [f"{var}_diff_lag_{lag}" for var in vars_short_lag for lag in short_lags] +
                [f"{var}_diff_lag_{lag}" for var in vars_long_lag for lag in long_lags]
            )

            # Add additional features
            merged_data['deaths_to_cases_ratio'] = np.where(
                merged_data['new_cases_smoothed'] == 0, 0,
                merged_data['new_deaths_smoothed'] / merged_data['new_cases_smoothed']
            )
            merged_data['interaction_term'] = merged_data['new_cases_smoothed'] * merged_data['Dummy_Variable']
            additional_features = ['deaths_to_cases_ratio', 'interaction_term']

            # Add additional features to independent variables
            independent_vars += additional_features

            # Drop rows with NaN values after differencing and lagging
            merged_data = merged_data.dropna(subset=independent_vars + ['target'])

            # Perform logistic regression
            st.write("Performing Logistic Regression...")
            logistic_model = perform_logistic_regression(merged_data, independent_vars)

            # Display Logistic Regression Results
            st.subheader("Logistic Regression Results")
            st.markdown("**Model Coefficients:**")
            coef_df = pd.DataFrame(logistic_model.coef_, columns=independent_vars)
            st.table(coef_df)

            st.markdown(f"**Intercept:** {logistic_model.intercept_}")

            st.markdown(f"**Accuracy on Test Data:** {logistic_model.score(merged_data[independent_vars], merged_data['target']):.4f}")

            from sklearn.metrics import classification_report
            y_true_logistic = merged_data['target']
            y_pred_logistic = logistic_model.predict(merged_data[independent_vars])
            logistic_report = classification_report(y_true_logistic, y_pred_logistic, output_dict=True)

            st.markdown("**Classification Report:**")
            logistic_report_df = pd.DataFrame(logistic_report).transpose()
            st.table(logistic_report_df)

            st.markdown("### Model Performance Plots")
            # Add ROC curve plot:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_true_logistic, logistic_model.predict_proba(merged_data[independent_vars])[:, 1])
            roc_auc = auc(fpr, tpr)

            st.write(f"**ROC AUC:** {roc_auc:.4f}")

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax.legend(loc='lower right')
            st.pyplot(fig)

            # Perform Random Forest classification
            st.write("Performing Random Forest Classification...")
            rf_model = perform_random_forest(merged_data, independent_vars)

            # Display the feature importance
            feature_importance = pd.DataFrame({
                'Feature': independent_vars,  
                'Importance': rf_model.feature_importances_
            })
            st.subheader("Random Forest Feature Importance")
            st.table(feature_importance)

            # Plot the feature importance
            feature_importance.plot(kind='barh', x='Feature', y='Importance', legend=False)
            plt.title('Random Forest Feature Importance')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.show()

            feature_importance.plot(kind='barh', x='Feature', y='Importance')
            st.pyplot(plt)



        except Exception as e:
            st.error(f"An error occurred: {e}")
            
    elif tab == "COVID-19 Map":
        st.write("Displaying the COVID-19 Interactive Map...")
        covid_data = clean_data(COVID_FILE)
        plot_covid_cases(covid_data)

        st.header("Interactive Time Series Exploration")
        plot_interactive_time_series(merged_data, date_column='date')

        # Scatter Plot Matrix
        st.subheader("Scatter Plot Matrix with Filters")
        plot_scatter_matrix(merged_data, events)

        # plot the heatmap
        plot_interactive_heatmap(merged_data, date_column='date', time_unit='month')

        

if __name__ == "__main__":
    main()
