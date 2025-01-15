import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path
import streamlit as st
import matplotlib.dates as mdates
import pandas as pd

# Plots smoothed COVID-19 cases globally over time
def plot_covid_cases(data):
    data = data[['location', 'new_cases_smoothed', 'date']].dropna() 
    data = data[data['new_cases_smoothed'] > 0] 

    fig = px.choropleth(
        data,
        locations="location",
        locationmode="country names",
        color="new_cases_smoothed",
        hover_name="location",
        animation_frame=data['date'].dt.strftime('%Y-%m-%d'),
        title="Global COVID-19 Cases Over Time",
        color_continuous_scale=px.colors.sequential.Plasma
    )
    st.plotly_chart(fig)


def plot_stock_with_events(data, events, output_path='./visualizations/stock_price_with_events.png'):
    data = data.drop_duplicates(subset=['date']).dropna(subset=['close'])
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))

    # Plot the stock price
    plt.plot(data['date'], data['close'], label='Stock Price', color='blue', linewidth=2)
    
    # Create a set to track already plotted event dates
    plotted_dates = set()

    # Plot events
    for event, date in events.items():
        event_date = pd.to_datetime(date)
        
        if event_date not in plotted_dates:
            plt.axvline(event_date, color='red', linestyle='--', label=event)
            plotted_dates.add(event_date)
    
    # Set the title and labels
    plt.title("Stock Price with Key Events", fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Stock Price (USD)', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.xticks(rotation=45)

    # Save and show 
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    st.pyplot(plt)


def visualize_covid_data(data, output_path='./visualizations/covid_trends.png'):
    covid_summary = data.groupby('date')[['new_cases']].sum().reset_index()
    covid_summary['new_cases_smoothed'] = covid_summary['new_cases'].rolling(7).mean()

    plt.figure(figsize=(14, 8))
    
    plt.plot(covid_summary['date'], covid_summary['new_cases'], label='New Cases (Raw)', color='red', linewidth=2)

    # Plot smoothed cases
    plt.plot(covid_summary['date'], covid_summary['new_cases_smoothed'], label='New Cases (Smoothed)', color='blue', linewidth=2)

    plt.title('Global COVID New Cases', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(alpha=0.3)
    
    # Improve readability
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)

    # Save and show
    plt.tight_layout()
    st.pyplot(plt) 


def plot_regression_results(coefficients, intercept, r2_score, feature_names, output_path='./visualizations/regression_results.png'):
    if len(coefficients) != len(feature_names):
        raise ValueError(f"Mismatch: {len(coefficients)} coefficients but {len(feature_names)} feature names provided.")
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, coefficients, color='skyblue', edgecolor='black')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title(f'Regression Coefficients (R² = {r2_score:.4f}, Intercept = {intercept:.4f})')
    plt.ylabel('Coefficient Value')
    plt.xlabel('Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt) 
