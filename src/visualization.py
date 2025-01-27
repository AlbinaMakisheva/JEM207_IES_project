import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path
import streamlit as st
import matplotlib.dates as mdates
import pandas as pd
import plotly.express as px

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

#Interactive Time Series 

def plot_interactive_time_series(data, date_column='date', default_vars=None):
    if default_vars is None:
        default_vars = ['new_cases_smoothed', 'new_deaths_smoothed', 'gdp_per_capita', 'stringency_index', 'new_vaccinations_smoothed']
    
    # Allow users to select variables to plot
    st.header("Interactive Time Series Exploration")
    st.write("Select variables to plot and explore trends interactively.")
    
    variables = st.multiselect(
        "Select Variables to Plot",
        data.columns.tolist(),
        default=default_vars
    )
    
    scale_type = st.radio("Select Scale Type:", ["Linear", "Logarithmic"], index=0)
    
    # Normalization option
    normalize = st.checkbox("Normalize Data (Min-Max Scaling)", value=False)
    
    if variables:
        # Normalize data
        if normalize:
            data[variables] = data[variables].apply(
                lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x
            )
        
        fig = px.line(
            data,
            x=date_column,
            y=variables,
            title="Interactive Time Series Exploration",
            labels={date_column: "Date"},
            markers=True
        )
        
        if scale_type == "Logarithmic":
            fig.update_layout(yaxis_type="log")
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Values",
            legend_title="Variables",
            template="plotly_white"
        )
        st.plotly_chart(fig)
    else:
        st.write("Please select at least one variable to plot.")

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


def plot_bubble_chart(data):
    """
    Creates an interactive bubble chart with temporal animation.
    """
    st.header("Pfizer Stock vs COVID-19 Metrics (Bubble Chart)")
    st.write("Visualizing stock price movements in relation to COVID-19 cases and vaccinations over time.")

    required_columns = ['date', 'Close', 'new_cases', 'total_vaccinations']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        return

    # Ensure 'date' is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'])

    # Create the bubble chart
    fig = px.scatter(
        data,
        x="new_cases",
        y="Close",
        size="total_vaccinations",
        color="Close",
        animation_frame=data['date'].dt.strftime('%Y-%m-%d'),  # Format dates for animation
        title="Bubble Chart: Pfizer Stock Price vs COVID-19 Cases with Vaccinations",
        labels={
            "new_cases": "New COVID-19 Cases",
            "Close": "Pfizer Stock Price (USD)",
            "total_vaccinations": "Total Vaccinations"
        },
        hover_data={"date": True},  # Show date in hover info
        color_continuous_scale=px.colors.sequential.Plasma
    )
    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='black')))
    fig.update_layout(template="plotly_white", xaxis_title="New COVID-19 Cases", yaxis_title="Pfizer Stock Price (USD)")
    st.plotly_chart(fig)


def plot_scatter_matrix(data, events):
    """
    Creates a scatter plot matrix with interactive filtering.
    """
    st.header("Scatter Matrix: Key COVID-19 and Stock Market Metrics")
    st.write("Explore how different variables relate to Pfizer's stock price.")

    variables = st.multiselect(
        "Select Variables to Include in the Scatter Matrix",
        data.columns.tolist(),
        default=['Close', 'new_cases', 'total_vaccinations', 'stringency_index']
    )

    event_filter = st.selectbox("Filter Data by Event", ["No Filter"] + list(events.keys()))
    if event_filter != "No Filter":
        event_date = pd.to_datetime(events[event_filter])
        time_window = st.slider("Time Window Around Event (Days)", 7, 90, 30)
        filtered_data = data[(data['date'] >= event_date - pd.Timedelta(days=time_window)) &
                             (data['date'] <= event_date + pd.Timedelta(days=time_window))]
    else:
        filtered_data = data

    if len(variables) >= 2:
        fig = px.scatter_matrix(
            filtered_data,
            dimensions=variables,
            color="Close",
            title="Scatter Plot Matrix: Stock & COVID-19 Metrics",
            labels={col: col.replace('_', ' ').title() for col in variables},
            template="plotly_white"
        )
        st.plotly_chart(fig)
    else:
        st.write("Please select at least two variables to plot.")


def plot_density_around_events(data, events):
    """
    Creates density plots of stock returns before, during, and after significant events.
    """
    st.header("Stock Returns Density Around Key Events")
    st.write("Visualizing Pfizer's stock return distribution around major COVID-19 events.")

    # Ensure the required column exists
    if 'daily_return' not in data.columns:
        st.error("Column 'daily_return' (representing stock returns) is missing from the data.")
        return

    event_filter = st.selectbox("Select Event to Analyze", list(events.keys()))
    if event_filter:
        event_date = pd.to_datetime(events[event_filter])
        time_window = st.slider("Time Window Around Event (Days)", 7, 90, 30)

        before_event = data[(data['date'] < event_date) & 
                            (data['date'] >= event_date - pd.Timedelta(days=time_window))]
        after_event = data[(data['date'] >= event_date) & 
                           (data['date'] <= event_date + pd.Timedelta(days=time_window))]

        # Ensure there is enough data for plotting
        if before_event.empty or after_event.empty:
            st.warning(f"Not enough data around the event '{event_filter}' to generate density plots.")
            return

        # Plot density plots
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.kdeplot(before_event['daily_return'], label='Before Event', fill=True, color='blue', alpha=0.5)
        sns.kdeplot(after_event['daily_return'], label='After Event', fill=True, color='red', alpha=0.5)

        plt.axvline(0, color='black', linestyle='--', linewidth=1)
        plt.title(f"Pfizer Stock Returns Before and After: {event_filter}", fontsize=16)
        plt.xlabel("Daily Returns", fontsize=14)
        plt.ylabel("Density", fontsize=14)
   

