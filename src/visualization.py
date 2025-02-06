import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import matplotlib.dates as mdates
import pandas as pd
import plotly.express as px
from sklearn.metrics import classification_report
from pathlib import Path


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


def display_classification_report(y_true, y_pred, model_name="Model"):
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write(f"{model_name} Classification Report:")
    st.dataframe(report_df)
    
 # Helper function for plotting regression coefficients
def plot_coefficients(coefficients_df, title="Feature Importance (Coefficients)"):
    fig, ax = plt.subplots(figsize=(8, 6))
    coefficients_df.plot.bar(x='Feature', y='Coefficient', legend=False, ax=ax)
    plt.title(title)
    plt.ylabel("Coefficient Value")
    plt.xlabel("Features")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)   
    
  # Helper function to ensure X and y have aligned rows
def align_data(X, y):
    """Align X and y by their indices to ensure compatibility."""
    aligned_data = pd.concat([X, y], axis=1).dropna()
    return aligned_data[X.columns], aligned_data[y.name]
            

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


def plot_residual_diagnostics(model, X, y, regression_name):
    try:
        # Ensure features match those used during fit
        if list(model.feature_names_in_) != list(X.columns):
            missing_features = set(model.feature_names_in_) - set(X.columns)
            unexpected_features = set(X.columns) - set(model.feature_names_in_)
            raise ValueError(
                f"Feature mismatch for {regression_name}:\n"
                f"Missing features: {missing_features}\n"
                f"Unexpected features: {unexpected_features}"
            )

        # Predict and calculate residuals
        predictions = model.predict(X)
        residuals = y - predictions

        # Plot Residual Diagnostics
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Residuals vs Fitted
        sns.scatterplot(x=predictions, y=residuals, ax=axs[0], color="blue", alpha=0.6)
        axs[0].axhline(0, color="red", linestyle="--", linewidth=1)
        axs[0].set_title("Residuals vs Fitted")
        axs[0].set_xlabel("Fitted Values")
        axs[0].set_ylabel("Residuals")

        # Histogram of Residuals
        sns.histplot(residuals, kde=True, bins=20, ax=axs[1], color="blue", alpha=0.6)
        axs[1].axhline(0, color="red", linestyle="--", linewidth=1)
        axs[1].set_title("Distribution of Residuals")

        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during residual diagnostics for {regression_name}: {e}")
     