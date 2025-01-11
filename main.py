import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from src.data_cleaning import clean_data
from src.data_merging import merge_data
import matplotlib.pyplot as plt 
from src.analysis import filter_data_around_events, perform_multiple_linear_regression, analyze_event_impact, prepare_binary_target, perform_logistic_regression, perform_extended_logistic_regression, perform_random_forest
from src.visualization import plot_covid_cases, plot_stock_with_events, visualize_covid_data, plot_regression_results
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
        
        plot_stock_with_events(merged_data, events)
        visualize_covid_data(covid_data)

    if tab == "Analysis":
        # Filter data around key events
        st.write("Filtering data around key events...")
        window_size = st.slider("Select window size around events (in months)", 1, 12, 3)
        filtered_data = filter_data_around_events(merged_data, events, window_months=window_size)

        try:
            # Perform multiple linear regression
            st.write("Performing Regression Analysis...")
            regression_model, r2_score = perform_multiple_linear_regression(
                filtered_data,
                dependent_var='daily_return',
                independent_vars=['new_vaccinations_smoothed', 'new_deaths_smoothed', 'new_cases_smoothed', 'Dummy_Variable']
            )

            # Display regression results
            st.subheader("Regression Results")
            st.markdown(f"**R² Score:** {r2_score:.4f}")
            st.markdown("**Coefficients:**")
            st.table(pd.DataFrame({
                'Feature': regression_model.feature_names_in_,
                'Coefficient': regression_model.coef_
            }))
            st.markdown(f"**Intercept:** {regression_model.intercept_:.4f}")

            # Plot regression results
            plot_regression_results(
                coefficients=regression_model.coef_,
                intercept=regression_model.intercept_,
                r2_score=r2_score,
                feature_names=regression_model.feature_names_in_
            )

            # Event impact analysis
            st.write("Analyzing Event Impact...")
            event_impact = analyze_event_impact(filtered_data)
            st.subheader("Event Impact on Stock Returns")
            st.write(event_impact)

            # Prepare binary target for logistic regression
            merged_data = prepare_binary_target(filtered_data, price_column='close') 
            independent_vars = ['new_vaccinations_smoothed', 'new_deaths_smoothed', 'new_cases_smoothed', 'Dummy_Variable']

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

            merged_data[extended_independent_vars] = merged_data[extended_independent_vars].replace([np.inf, -np.inf], np.nan).fillna(0)

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

            extended_report = classification_report(y_test, extended_logistic_model.predict(X_test), output_dict=True, zero_division=0)
            extended_report_df = pd.DataFrame(extended_report).transpose()
            st.markdown("**Classification Report (Extended):**")
            st.table(extended_report_df)

            st.markdown("### Model Performance Plots")

            # Calculate ROC Curve
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

        except KeyError as e:
            st.error(f"Analysis error: {e}")
            
    elif tab == "COVID-19 Map":
        st.write("Displaying the COVID-19 Interactive Map...")
        covid_data = clean_data(COVID_FILE)
        plot_covid_cases(covid_data)
        

if __name__ == "__main__":
    main()
