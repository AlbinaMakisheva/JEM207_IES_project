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
        title="Cases Over Time",
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
     

def plot_interactive_time_series(data, date_column='date', default_vars=None):
    if default_vars is None:
        default_vars = ['new_cases_smoothed', 'new_deaths_smoothed', 'gdp_per_capita', 'stringency_index', 'new_vaccinations_smoothed']
    
    # Allow users to select variables to plot
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