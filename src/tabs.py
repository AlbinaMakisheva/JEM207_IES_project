import streamlit as st
from src.process_data import prepare_data_for_regressions
from src.visualization import plot_stock_with_events, visualize_covid_data
from src.texts import limitations_text, filtering_text, covid_cases_analysis_text, stock_price_analysis_text, introduction_text
from src.main_functions import filter_data_step, perform_autocorrelation_analysis, categorize_and_process_features, check_stationarity_and_multicollinearity, perform_linear_regressions, perform_residual_diagnostics, perform_heteroscedasticity_analysis, perform_logistic_regression_analysis

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
    st.header("Analysis")

    # Filter Data
    filtered_data = filter_data_step(merged_data, events)

    # Perform Autocorrelation Analysis
    if st.button("Perform Autocorrelation Analysis and Analyze Multicollinearity"):
        try:
            perform_autocorrelation_analysis(merged_data)
            autocorrelation_vars = categorize_and_process_features(merged_data)
            check_stationarity_and_multicollinearity(merged_data, autocorrelation_vars)
        except Exception as e:
            st.error(f"Error during analysis: {e}")

    # Prepare Data for Regression
    prepared_data, independent_vars_sets = prepare_data_for_regressions(filtered_data)

    if st.button("Perform Linear Regressions"):
        try:
            perform_linear_regressions(prepared_data, independent_vars_sets)
        except Exception as e:
            st.error(f"Error during regression analysis: {e}")

    if st.button("Perform Residual Diagnostics"):
        try:
            perform_residual_diagnostics(prepared_data, independent_vars_sets)
        except Exception as e:
            st.error(f"Error during residual diagnostics: {e}")

    if st.button("Perform Heteroscedasticity Analysis"):
        try:
            perform_heteroscedasticity_analysis(prepared_data, independent_vars_sets)
        except Exception as e:
            st.error(f"Error during heteroscedasticity analysis: {e}")

    if st.button("Perform Logistic Regression Analysis"):
        try:
            perform_logistic_regression_analysis(prepared_data)
        except Exception as e:
            st.error(f"Error during logistic regression analysis: {e}")

    # Limitations
    st.header("Limitations of the project")
    st.write(limitations_text())

