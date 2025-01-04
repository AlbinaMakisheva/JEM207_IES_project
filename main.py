import os
from src.data_cleaning import clean_data
from src.data_merging import merge_data
from src.analysis import filter_data_around_events, perform_multiple_linear_regression, analyze_event_impact, prepare_binary_target, perform_logistic_regression
from src.visualization import plot_covid_cases, plot_stock_with_events, visualize_covid_data, plot_regression_results
from src.data_fetching import fetch_covid_data, fetch_stock_data


COVID_FILE = 'data/raw/covid_data.csv'
STOCK_FILE = 'data/raw/pfizer_stock.csv'

def main():
    if not os.path.exists(COVID_FILE):
        print("Fetching COVID data...")
        fetch_covid_data(COVID_FILE)
        
    if not os.path.exists(STOCK_FILE):
        print("Fetching stock data...")
        fetch_stock_data(STOCK_FILE)

    # Key events
    events = {
        "WHO Declares Pandemic": "2020-03-11",
        "First Vaccine": "2020-12-08",
        "Vaccination Threshold Reached (85%)": "2021-07-30",
        "Relaxation of Lockdowns": "2022-05-01",  
        "Omicron-Specific Vaccine Approval": "2022-09-01", 
        "China Eases Zero-COVID Policy": "2023-01-01", 
    }
    
    # Clean data
    covid_data = clean_data(COVID_FILE)
    stock_data = clean_data(STOCK_FILE, is_stock=True)
    
    # Merge datasets
    merged_data = merge_data(covid_data, stock_data, events)

    # Filter data around key events
    filtered_data = filter_data_around_events(merged_data, events)

    try:
        regression_model, r2_score = perform_multiple_linear_regression(
            filtered_data,
            dependent_var='daily_return',
            independent_vars=[
                'new_vaccinations_smoothed_per_million',
                'new_cases',
                'Dummy_Variable',
                'stringency_index',
                'new_cases_per_million',
                'total_vaccinations_per_hundred',
                'positive_rate',
                'gdp_per_capita',
                'reproduction_rate'
            ]
        )
        
        plot_regression_results(
            coefficients=regression_model.coef_,
            intercept=regression_model.intercept_,
            r2_score=r2_score,
            feature_names=regression_model.feature_names_in_
        )
        event_impact = analyze_event_impact(filtered_data)

        # Logistic regression analysis
        merged_data = prepare_binary_target(merged_data, price_column='close')  # Prepare binary target
        independent_vars = ['new_vaccinations_smoothed', 'new_deaths_smoothed', 'new_cases_smoothed', 'Dummy_Variable']
        logistic_model = perform_logistic_regression(merged_data, independent_vars)

    except KeyError as e:
        print(f"Analysis error: {e}")
        

    # plot_stock_with_events(merged_data, events)
    # visualize_covid_data(covid_data)
    # plot_covid_cases(merged_data)
if __name__ == "__main__":
    main()
