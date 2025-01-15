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
from src.analysis import filter_data_around_events, perform_multiple_linear_regression, analyze_event_impact, prepare_binary_target, perform_logistic_regression, perform_extended_logistic_regression, perform_random_forest
from src.visualization import plot_covid_cases, plot_stock_with_events, visualize_covid_data, plot_regression_results, plot_roc_curve, display_classification_report, plot_feature_importance


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
        
    plot_stock_with_events(merged_data, events)
    visualize_covid_data(covid_data)



def analysis_tab(merged_data, events):
    # Filter data around key events
    st.write("Filtering data around key events...")
    window_size = st.slider("Select window size around events (in months)", 1, 12, 3)
    filtered_data = filter_data_around_events(merged_data, events, window_months=window_size)
    

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
    except ValueError as e:
        st.error(f"ValueError encountered: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
