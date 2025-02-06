import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve, classification_report
from src.process_data import filter_data_around_events, prepare_binary_target, process_data_for_regressions, apply_differencing, categorize_by_autocorrelation
from src.analysis import perform_extended_logistic_regression, perform_and_display_regression, plot_residual_diagnostics_for_model, run_regression_analysis, plot_lag_correlations, calculate_feature_importance, check_stationarity, reduce_multicollinearity
from src.visualization import plot_residual_diagnostics, plot_stock_with_events, visualize_covid_data, plot_roc_curve, display_classification_report, plot_interactive_heatmap
from src.texts import limitations_text, ext_logReg_coef_text, ext_logRed_text, logReg_coef_text, logReg_text, heteroscedasticity_text, residual_text, coefficients_text_secondLR, coefficients_text_firstLR, autocorrelation_text, filtering_text, covid_cases_analysis_text, covid_cases_analysis_text, stock_price_analysis_text, introduction_text

def introduction_tab(merged_data, events, covid_data):
    st.header("Introduction")
    st.write(introduction_text())
    
    # Stock Price with Key Events
    st.write("### Stock Price with Key Events")
    plot_stock_with_events(merged_data, events)
    st.write(stock_price_analysis_text())

    # Global COVID-19 New Cases
    st.write("### Global COVID-19 New Cases")
    visualize_covid_data(covid_data)
    st.write(covid_cases_analysis_text())



def analysis_tab(merged_data, events):
    """Main function to perform analysis."""
    # Step 1: Filter Data
    filtered_data = filter_data_step(merged_data, events)

    # Step 2: Perform Autocorrelation Analysis
    perform_autocorrelation_analysis(merged_data)

    # Step 3: Categorize Variables by Autocorrelation and Process Features
    critical_high_autocorrelation_vars = categorize_and_process_features(merged_data)

    # Step 4: Check Stationarity and Multicollinearity
    check_stationarity_and_multicollinearity(merged_data, critical_high_autocorrelation_vars)

    # Step 5: Prepare Data for Regression
    filtered_data, independent_vars_sets = prepare_data_for_regressions(filtered_data)

    # Step 6: Perform Linear Regressions
    perform_linear_regressions(filtered_data, independent_vars_sets)

    # Step 7: Residual Diagnostics
    perform_residual_diagnostics(filtered_data, independent_vars_sets)

    # Step 8: Heteroscedasticity Analysis
    perform_heteroscedasticity_analysis(filtered_data, independent_vars_sets)

    # Step 9: Logistic Regression Analysis
    perform_logistic_regression_analysis(filtered_data)


def filter_data_step(merged_data, events):
    """Filters data around key events."""
    st.write("Filtering data around key events...")
    window_size = st.slider("Select window size around events (in months)", 1, 12, 3)
    filtered_data = filter_data_around_events(merged_data, events, window_months=window_size)
    st.write(filtering_text())
    return filtered_data


def perform_autocorrelation_analysis(merged_data):
    """Performs autocorrelation analysis and categorizes variables."""
    try:
        st.write("Performing autocorrelation analysis...")
        plot_lag_correlations(merged_data, lag=1)
        st.write(autocorrelation_text())
    except KeyError as e:
        st.error(f"Error during autocorrelation analysis: {e}")


def categorize_and_process_features(merged_data):
    """Categorizes variables by autocorrelation and processes them."""
    try:
        st.write("Categorizing variables by autocorrelation...")
        autocorrelation_categories = categorize_by_autocorrelation(merged_data, lag=1)

        high_vars, moderate_vars, low_vars = (
            autocorrelation_categories['high'],
            autocorrelation_categories['moderate'],
            autocorrelation_categories['low'],
        )

        st.write("### Autocorrelation Categories")
        st.write("#### High Autocorrelation Variables:", high_vars)
        st.write("#### Moderate Autocorrelation Variables:", moderate_vars)
        st.write("#### Low Autocorrelation Variables:", low_vars)

        st.write("Calculating feature importance...")
        feature_importance = calculate_feature_importance(merged_data, target_var='daily_return')
        st.table(feature_importance)

        top_vars = feature_importance['Feature'].head(10).tolist()
        critical_high_vars = [var for var in high_vars if var in top_vars]

        st.write("Applying differencing to critical variables...")
        merged_data = apply_differencing(merged_data, critical_high_vars)
        st.write("Differencing applied:", [f'diff_{var}' for var in critical_high_vars])

        return critical_high_vars
    except KeyError as e:
        st.error(f"Error during analysis: {e}")
        return []


def check_stationarity_and_multicollinearity(merged_data, critical_high_vars):
    """Checks stationarity and multicollinearity."""
    try:
        test_vars = [f'diff_{var}' for var in critical_high_vars]

        # Check Stationarity on selected variables: Augmented Dickey-Fuller Test
        # st.write("Perfoming Stationarity...")
        # stationarity_results = check_stationarity(merged_data[test_vars])
        # st.table(stationarity_results)

        st.write("Analyzing multicollinearity (VIF)...")
        vif_results = reduce_multicollinearity(merged_data[test_vars], threshold=10)
        st.table(vif_results)
    except KeyError as e:
        st.error(f"Error during stationarity and multicollinearity analysis: {e}")


def prepare_data_for_regressions(filtered_data):
    """Prepares data for regression analysis."""
    short_lags, long_lags = [1], [180]
    filtered_data = process_data_for_regressions(filtered_data, short_lags, long_lags)

    reg1_vars = (
        [f"{var}_diff_lag_{lag}" for var in ['new_cases_smoothed', 'new_deaths_smoothed'] for lag in short_lags] +
        [f"{var}_diff_lag_{lag}" for var in ['vaccination_signal', 'reproduction_rate', 'new_vaccinations_smoothed', 'Dummy_Variable'] for lag in long_lags] +
        ['new_cases_dummy_interaction', 'new_deaths_dummy_interaction']
    )
    
    reg2_vars = (
        [f"reproduction_rate_vaccinations_diff_lag_{lag}" for lag in short_lags] +
        [f"{var}_diff_lag_{lag}" for var in ['vaccination_signal', 'Dummy_Variable'] for lag in long_lags] +
        ['deaths_to_cases_ratio', 'new_cases_dummy_interaction', 'new_deaths_dummy_interaction']
    )

    independent_vars = ['new_cases_smoothed', 'Dummy_Variable', 'stringency_index', 'new_cases_dummy_interaction', 'total_vaccination_rate', 'female_smokers_rate']

    return filtered_data, {'reg1': reg1_vars, 'reg2': reg2_vars, 'independent': independent_vars}


def perform_linear_regressions(filtered_data, independent_vars_sets):
    """Performs linear regression analysis."""
    st.header("Add Lag Variables")
    st.write("""
        Given the high correlation in most of our variables and the characteristics of our analysis, we decided to create lag variables. Lag variables account for the time delay between a predictor variable and its effect on the dependent variable (stock returns). We believe that this is highly relevant for variables like vaccinations that take time to impact COVID-19 metrics, which in turn may influence Pfizer's stock performance.
    """)

    try:
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

        perform_and_display_regression(filtered_data, 'daily_return', independent_vars_sets['reg2'])
        st.write(coefficients_text_firstLR())

        st.header("Second Linear Regression")
        st.latex(r"""
        \mathrm{new\_deaths\_smoothed} = \beta_0 + \beta_1 (\mathrm{new\_cases\_smoothed}) + 
        \beta_2 (\mathrm{Dummy\_Variable}) + 
        \beta_3 (\mathrm{stringency\_index}) + 
        \beta_4 (\mathrm{new\_cases\_dummy\_interaction}) + 
        \beta_5 (\mathrm{total\_vaccination\_rate}) + 
        \beta_6 (\mathrm{female\_smokers\_rate}) + \epsilon
        """)

        perform_and_display_regression(filtered_data, 'new_deaths_smoothed', independent_vars_sets['independent'])
        st.write(coefficients_text_secondLR())

    except KeyError as e:
        st.error(f"Error during regression analysis: {e}")


def perform_residual_diagnostics(filtered_data, independent_vars_sets):
    """Performs residual diagnostics for regression models."""
    st.header("Residual Diagnostics")
    plot_residual_diagnostics_for_model(filtered_data, independent_vars_sets['reg2'], 'daily_return', "First Regression")
    plot_residual_diagnostics_for_model(filtered_data, independent_vars_sets['independent'], 'new_deaths_smoothed', "Second Regression")

    #Observations from the Residual Diagnostics
    st.write("Observations from the Residual Diagnostics...")
    st.write(residual_text())


def perform_heteroscedasticity_analysis(filtered_data, independent_vars_sets):
    """Performs heteroscedasticity analysis."""
    st.header("Heteroscedasticity Analysis")
    st.write("Testing for heteroscedasticity in regression models...")

    run_regression_analysis(filtered_data, 'daily_return', independent_vars_sets['reg2'], "First Regression")
    run_regression_analysis(filtered_data, 'new_deaths_smoothed', independent_vars_sets['independent'], "Second Regression")

    st.write(heteroscedasticity_text())


def perform_logistic_regression_analysis(filtered_data):
    """Performs logistic regression analysis on the given dataset."""
    try:
        # Define lag periods
        short_lags, long_lags = [1], [180]

        # Preprocess data
        filtered_data = process_data_for_regressions(filtered_data, short_lags, long_lags)
        filtered_data = prepare_binary_target(filtered_data, price_column='close')

        # Ensure deaths_to_cases_ratio exists before using it
        filtered_data['deaths_to_cases_ratio'] = np.where(
            filtered_data['new_cases_smoothed'] == 0, 0,
            filtered_data['new_deaths_smoothed'] / filtered_data['new_cases_smoothed']
        )

        # Interaction term: cases * dummy variable
        filtered_data['interaction_term'] = (
            filtered_data['new_cases_smoothed'] * filtered_data['Dummy_Variable']
        ).fillna(0)

        # Standard Logistic Regression Variables
        standard_vars = [
            'new_cases_smoothed', 'new_deaths_smoothed', 'new_vaccinations_smoothed',
            'Dummy_Variable', 'deaths_to_cases_ratio', 'interaction_term'
        ]

        # Extended Logistic Regression Variables
        extended_vars = [
            'new_cases_smoothed_diff_lag_1', 'new_deaths_smoothed_diff_lag_1',
            'new_vaccinations_smoothed_diff_lag_180', 'Dummy_Variable_diff_lag_180',
            'deaths_to_cases_ratio', 'interaction_term'
        ]

        # Drop rows with missing values in relevant columns
        filtered_data.dropna(subset=standard_vars + ['target'], inplace=True)
        filtered_data.dropna(subset=extended_vars + ['target'], inplace=True)

        ## Standard Logistic Regression
        st.subheader("Standard Logistic Regression")
        st.latex(r"""
            P(\text{target} = 1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \text{new\_cases\_smoothed} + 
            \beta_2 \text{new\_deaths\_smoothed} + 
            \beta_3 \text{new\_vaccinations\_smoothed} + 
            \beta_4 \text{Dummy\_Variable} + 
            \beta_5 \text{deaths\_to\_cases\_ratio} + 
            \beta_6 \text{interaction\_term})} }
        """)

        std_log_model, std_acc, std_fpr, std_tpr, std_roc_auc, std_coeffs, _ = (
            perform_extended_logistic_regression(filtered_data, standard_vars)
        )

        st.write(f"Accuracy: {std_acc}")
        display_classification_report(filtered_data['target'], std_log_model.predict(filtered_data[standard_vars]), 
                                      model_name="Standard Logistic Regression")
        plot_roc_curve(std_fpr, std_tpr, std_roc_auc, title="Standard Logistic Regression ROC Curve")
        st.write(logReg_text())

        # Display coefficients
        st.write("Standard Logistic Regression Coefficients:")
        st.table(pd.DataFrame({'Feature': standard_vars, 'Coefficient': std_log_model.coef_[0]}))

        st.write(logReg_coef_text())

        ## Extended Logistic Regression
        st.subheader("Extended Logistic Regression (Differencing & Lagging)")
        st.latex(r"""
            P(\text{target} = 1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \Delta \text{new\_cases\_smoothed}_{t-1} + 
            \beta_2 \Delta \text{new\_deaths\_smoothed}_{t-1} + 
            \beta_3 \Delta \text{new\_vaccinations\_smoothed}_{t-180} + 
            \beta_4 \Delta \text{Dummy\_Variable}_{t-180} + 
            \beta_5 \text{deaths\_to\_cases\_ratio} + 
            \beta_6 \text{interaction\_term})} }
        """)

        ext_log_model, ext_acc, ext_fpr, ext_tpr, ext_roc_auc, ext_coeffs, _ = (
            perform_extended_logistic_regression(filtered_data, extended_vars)
        )

        st.write(f"Accuracy: {ext_acc}")
        display_classification_report(filtered_data['target'], ext_log_model.predict(filtered_data[extended_vars]), 
                                      model_name="Extended Logistic Regression (Differencing & Lagging)")
        plot_roc_curve(ext_fpr, ext_tpr, ext_roc_auc, title="Extended Logistic Regression ROC Curve")
        st.write(ext_logRed_text())

        # Display coefficients
        st.write("Extended Logistic Regression Coefficients:")
        st.table(pd.DataFrame({'Feature': extended_vars, 'Coefficient': ext_log_model.coef_[0]}))

        st.write(ext_logReg_coef_text())

        st.write(
            "Despite the modifications (differencing & lagging), "
            "this model does not significantly outperform the standard logistic regression."
        )

        ## Limitations
        st.header("Limitations of the project")
        st.write(limitations_text())

    except KeyError as e:
        st.error(f"KeyError encountered: {e}")
