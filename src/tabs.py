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
from src.visualization import plot_coefficients, plot_residual_diagnostics, plot_covid_cases, plot_stock_with_events, visualize_covid_data, plot_regression_results, plot_roc_curve, display_classification_report, plot_feature_importance, plot_interactive_time_series, plot_interactive_heatmap
from src.texts import limitations_text, ext_logReg_coef_text, ext_logRed_text, logReg_coef_text, logReg_text, heteroscedasticity_text, residual_text, coefficients_text_secondLR, coefficients_text_firstLR, autocorrelation_text, filtering_text, covid_cases_analysis_text, covid_cases_analysis_text, stock_price_analysis_text, introduction_text

def introduction_tab(merged_data, events, covid_data):
    st.header("Introduction")
    st.write(introduction_text())
    
    # Graph 1: Stock Price with Key Events
    st.write("### Stock Price with Key Events")
    plot_stock_with_events(merged_data, events)
    st.write(stock_price_analysis_text())

    # Graph 2: Global COVID-19 New Cases
    st.write("### Global COVID-19 New Cases")
    visualize_covid_data(covid_data)
    st.write(covid_cases_analysis_text())



def analysis_tab(merged_data, events):
    # Filter data around key events
    st.write("Filtering data around key events...")
    window_size = st.slider("Select window size around events (in months)", 1, 12, 3)
    filtered_data = filter_data_around_events(merged_data, events, window_months=window_size)
    
    st.write(filtering_text())
    
    try:
        # Perform autocorrelation analysis
        st.write("Performing autocorrelation analysis...")
        plot_lag_correlations(merged_data, lag=1)

        st.write(autocorrelation_text)
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

        st.latex(r"""
        \text{daily return} = \beta_0 + \beta_1 (\mathrm{reproduction\_rate\_vaccinations\_diff\_lag\_1}) 
        + \beta_2 (\Delta \mathrm{vaccination\_signal\_diff\_lag\_180}) 
        + \beta_3 (\mathrm{Dummy\_Variable\_diff\_lag\_180}) 
        + \beta_4 (\mathrm{deaths\_to\_cases\_ratio}) 
        + \beta_5 (\mathrm{new\_cases\_dummy\_interaction}) 
        + \beta_6 (\mathrm{new\_deaths\_dummy\_interaction}) 
        + \epsilon
        """)

        regression_model = perform_and_display_regression(filtered_data, dependent_var='daily_return', independent_vars=reg2_independent_vars)

        st.write(coefficients_text_firstLR())
        
        # Second Linear Regression
        st.header("Second Linear Regression")

        st.latex(r"""
        \mathrm{new\_deaths\_smoothed} = \beta_0 + \beta_1 (\mathrm{new\_cases\_smoothed}) + 
        \beta_2 (\mathrm{Dummy\_Variable}) + 
        \beta_3 (\mathrm{stringency\_index}) + 
        \beta_4 (\mathrm{new\_cases\_dummy\_interaction}) + 
        \beta_5 (\mathrm{total\_vaccination\_rate}) + 
        \beta_6 (\mathrm{female\_smokers\_rate}) + \epsilon
        """)

        regression_model = perform_and_display_regression(filtered_data, dependent_var='new_deaths_smoothed', independent_vars=independent_vars)

        st.write(coefficients_text_secondLR())

        # Residual Diagnostics for each regression
        st.header("Residual Diagnostics")
        st.write("The purpose of this analysis is to visualize residuals to check the goodness-of-fit of regression models")

        plot_residual_diagnostics_for_model(filtered_data, reg2_independent_vars, 'daily_return', model_name="First Regression")
        plot_residual_diagnostics_for_model(filtered_data, independent_vars, 'new_deaths_smoothed', model_name="Second Regression")

        #Observations from the Residual Diagnostics
        st.write("Observations from the Residual Diagnostics...")

        st.write(residual_text())

    except KeyError as e:
        st.error(f"Error during analysis: {e}")

    
        # Heteroscedasticity Analysis
    st.header("Heteroscedasticity Analysis")
    st.write("Testing for heteroscedasticity in the regression models and applying corrections if needed...")


    # Running regression analyses
    run_regression_analysis(filtered_data, 'daily_return', reg2_independent_vars, "First Regression")

    run_regression_analysis(filtered_data, 'new_deaths_smoothed', independent_vars, "Second Regression")

    
    # Interpretation of heteroscedasticity results
    st.write(heteroscedasticity_text())
    
    try:
        
        # Define short and long lag periods
        short_lags = [1]  # 1 time-step lag
        long_lags = [180]  # 180 time-step lag

        # Apply preprocessing function from the analysis folder
        filtered_data = process_data_for_regressions(filtered_data, short_lags, long_lags)

        # Ensure binary target is created
        filtered_data = prepare_binary_target(filtered_data, price_column='close')

        # Ensure deaths_to_cases_ratio exists **before** using it
        filtered_data['deaths_to_cases_ratio'] = np.where(
            filtered_data['new_cases_smoothed'] == 0, 0,
            filtered_data['new_deaths_smoothed'] / filtered_data['new_cases_smoothed']
        )

        # Ensure interaction_term exists **after** processing
        filtered_data['interaction_term'] = filtered_data['new_cases_smoothed'] * filtered_data['Dummy_Variable']
        filtered_data['interaction_term'].fillna(0, inplace=True)

        # Print available columns to verify
        #st.write("Columns in filtered_data before regression:", filtered_data.columns.tolist())

        # Standard Logistic Regression Independent Variables
        standard_independent_vars = [
            'new_cases_smoothed', 'new_deaths_smoothed', 'new_vaccinations_smoothed', 'Dummy_Variable',
            'deaths_to_cases_ratio', 'interaction_term'
        ]

        # Extended Logistic Regression (using differencing & lagging)
        extended_independent_vars = [
            'new_cases_smoothed_diff_lag_1',
            'new_deaths_smoothed_diff_lag_1',
            'new_vaccinations_smoothed_diff_lag_180',
            'Dummy_Variable_diff_lag_180',
            'deaths_to_cases_ratio',
            'interaction_term'
        ]

        # Drop rows with missing values in the required columns **AFTER defining all variables**
        filtered_data = filtered_data.dropna(subset=standard_independent_vars + ['target'])
        filtered_data = filtered_data.dropna(subset=extended_independent_vars + ['target'])

        ### **Standard Logistic Regression**
        st.subheader("Standard Logistic Regression")
        st.latex(r"""
            P(\text{target} = 1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \text{new\_cases\_smoothed} + 
            \beta_2 \text{new\_deaths\_smoothed} + 
            \beta_3 \text{new\_vaccinations\_smoothed} + 
            \beta_4 \text{Dummy\_Variable} + 
            \beta_5 \text{deaths\_to\_cases\_ratio} + 
            \beta_6 \text{interaction\_term})} }
        """)

        std_log_model, std_acc, std_fpr, std_tpr, std_roc_auc, std_coeffs, _ = perform_extended_logistic_regression(
            filtered_data, standard_independent_vars
        )

        st.write(f"Standard Logistic Regression Accuracy: {std_acc}")

        # Classification report for standard logistic regression
        std_y_true = filtered_data['target']
        std_y_pred = std_log_model.predict(filtered_data[standard_independent_vars])
        display_classification_report(std_y_true, std_y_pred, model_name="Standard Logistic Regression")
        plot_roc_curve(std_fpr, std_tpr, std_roc_auc, title="Standard Logistic Regression ROC Curve")

        st.write(logReg_text())
                 
        # Coefficients for standard logistic regression
        st.write("Standard Logistic Regression Coefficients:")
        std_coef_df = pd.DataFrame({
            'Feature': standard_independent_vars,
            'Coefficient': std_log_model.coef_[0]
        })
        st.table(std_coef_df)

        st.write(" Regarding the interpretation of the coefficients..")

        st.write(logReg_coef_text())
         
        #**Extended Logistic Regression (Differencing & Lagging)**
        st.subheader("Extended Logistic Regression (Differencing & Lagging)")
        st.latex(r"""
            P(\text{target} = 1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \Delta \text{new\_cases\_smoothed}_{t-1} + 
            \beta_2 \Delta \text{new\_deaths\_smoothed}_{t-1} + 
            \beta_3 \Delta \text{new\_vaccinations\_smoothed}_{t-180} + 
            \beta_4 \Delta \text{Dummy\_Variable}_{t-180} + 
            \beta_5 \text{deaths\_to\_cases\_ratio} + 
            \beta_6 \text{interaction\_term})} }
        """)

        ext_log_model, ext_acc, ext_fpr, ext_tpr, ext_roc_auc, ext_coeffs, _ = perform_extended_logistic_regression(
            filtered_data, extended_independent_vars
        )

        st.write(f"Extended Logistic Regression Accuracy: {ext_acc}")

        # Classification report for extended logistic regression
        ext_y_true = filtered_data['target']
        ext_y_pred = ext_log_model.predict(filtered_data[extended_independent_vars])
        display_classification_report(ext_y_true, ext_y_pred, model_name="Extended Logistic Regression (Differencing & Lagging)")
        plot_roc_curve(ext_fpr, ext_tpr, ext_roc_auc, title="Extended Logistic Regression (Differencing & Lagging) ROC Curve")

        st.write(ext_logRed_text())


        # Coefficients for extended logistic regression
        st.write("Extended Logistic Regression (Differencing & Lagging) Coefficients:")
        ext_coef_df = pd.DataFrame({
            'Feature': extended_independent_vars,
            'Coefficient': ext_log_model.coef_[0]
        })
        st.table(ext_coef_df)

        st.write(" Regarding the interpretation of the coefficients..")

        st.write(ext_logReg_coef_text())
                 
        st.write("Despite the modifications (differencing & lagging), this model does not significantly outperform the standard logistic regression, with similar accuracy and ROC AUC values.")

        st.header("Limitations of the project")
        
        st.write(limitations_text())

    except KeyError as e:
        st.error(f"KeyError encountered: {e}")
