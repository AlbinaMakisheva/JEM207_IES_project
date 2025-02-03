import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path
import streamlit as st
import matplotlib.dates as mdates
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
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

def plot_roc_curve(fpr, tpr, roc_auc, title="ROC Curve"):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    st.pyplot(fig)

def plot_feature_importance(feature_importance_df, title="Feature Importance"):
    feature_importance_df.plot(kind='barh', x='Feature', y='Importance', legend=False)
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()
    st.pyplot(plt)


def display_classification_report(y_true, y_pred, model_name="Model"):
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write(f"{model_name} Classification Report:")
    st.dataframe(report_df)
    
    
    
    
    
    
    
    
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
        
def plot_scatter_matrix(data, events):
    """
    Creates a scatter plot matrix with interactive filtering and better scaling. 
        
    """
    st.header("Enhanced Scatter Matrix: Stock & COVID-19 Metrics")
    st.write("""Explore how different variables relate to Pfizer's stock price. Each subplot compares two variables: If most points form a line, they have a strong correlation. If points are scattered randomly, the variables have little/no correlation.""")

    # Default key variables
    default_vars = ['daily_return', 'new_cases', 'total_vaccinations', 'stringency_index']
    default_vars = [var for var in default_vars if var in data.columns]

    variables = st.multiselect(
        "Select Variables to Include in the Scatter Matrix",
        data.select_dtypes(include=['number']).columns.tolist(),
        default=default_vars
    )

    # Apply event-based filtering
    event_filter = st.selectbox("Filter Data by Event", ["No Filter"] + list(events.keys()))
    if event_filter != "No Filter":
        event_date = pd.to_datetime(events[event_filter])
        time_window = st.slider("Time Window Around Event (Days)", 7, 90, 30)
        filtered_data = data[(data['date'] >= event_date - pd.Timedelta(days=time_window)) &
                             (data['date'] <= event_date + pd.Timedelta(days=time_window))]
    else:
        filtered_data = data

    scaler = MinMaxScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(filtered_data[variables]), columns=variables)

    if len(variables) >= 2:
        fig = px.scatter_matrix(
            scaled_data,
            dimensions=variables,
            color=filtered_data['daily_return'] if "daily_return" in filtered_data.columns else variables[0],
            title="Enhanced Scatter Plot Matrix: Stock & COVID-19 Metrics",
            labels={col: col.replace('_', ' ').title() for col in variables},
            template="plotly_white",
            color_continuous_scale="Viridis"  
        )
        st.plotly_chart(fig)
    else:
        st.warning("Please select at least two variables to plot.")


def plot_interactive_heatmap(data, date_column='date', time_unit='month'):
    """
    Creates an enhanced interactive heatmap with a time slider to explore relationships over time.
    """
    st.header("Interactive Heatmap: Stock Returns vs COVID-19 Metrics Over Time")
    st.write("Explore how stock returns correlate with key COVID-19 metrics over different time periods.")

    if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
        data[date_column] = pd.to_datetime(data[date_column])

    # Aggregate by selected time unit
    if time_unit == 'month':
        data['time_unit'] = data[date_column].dt.to_period('M').astype(str)  # Convert Period to string
    elif time_unit == 'quarter':
        data['time_unit'] = data[date_column].dt.to_period('Q').astype(str)  # Convert Period to string
    elif time_unit == 'year':
        data['time_unit'] = data[date_column].dt.to_period('Y').astype(str)  # Convert Period to string
    else:
        st.error("Invalid time unit selected. Please choose 'month', 'quarter', or 'year'.")
        return
    
    numeric_data = data.select_dtypes(include=['number'])

    # Allow users to select variables
    available_variables = numeric_data.columns.tolist()
    selected_variables = st.multiselect(
        "Select Variables for Heatmap",
        available_variables,
        default=['daily_return', 'new_cases', 'new_deaths', 'total_vaccinations', 'stringency_index']
    )

    if 'daily_return' not in selected_variables:
        selected_variables.append('daily_return')  

    correlation_data = []
    
    # Compute correlations for each time period
    for time_period, group in numeric_data.groupby(data['time_unit']):
        if len(group) < 2:
            st.warning(f"Skipping time period {time_period}: Not enough data for correlation.")
            continue
        
        group = group[selected_variables]
        corr_matrix = group.corr()[['daily_return']].reset_index()  
        corr_matrix['time_unit'] = time_period  
        correlation_data.append(corr_matrix)

    if not correlation_data:
        st.error("No valid correlation data found. Ensure sufficient data for each time period.")
        return
    
    heatmap_data = pd.concat(correlation_data, ignore_index=True)

    heatmap_pivot = heatmap_data.pivot(index='time_unit', columns='index', values='daily_return')

    # Sort variables by correlation strength
    variable_order = heatmap_pivot.abs().mean(axis=0).sort_values(ascending=False).index
    heatmap_pivot = heatmap_pivot[variable_order]

    fig = px.imshow(
        heatmap_pivot,
        labels=dict(x="Variables", y="Time Period", color="Correlation"),
        title="Enhanced Correlation of Stock Returns with COVID-19 Metrics Over Time",
        aspect="auto",
        color_continuous_scale="Cividis", 
        zmin=-0.5,  
        zmax=0.5
    )

    fig.update_layout(
        xaxis=dict(tickangle=45),  
        template="plotly_white",
        font=dict(size=12)
    )

    st.plotly_chart(fig)