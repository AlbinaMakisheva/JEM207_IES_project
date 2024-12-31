import pandas as pd
import plotly.express as px

def plot_cases_country(data):
    data['date'] = pd.to_datetime(data['date'])

    # Clean data
    data = data[['location', 'new_cases_smoothed', 'date']].dropna() 
    data = data[data['new_cases_smoothed'] > 0] 

    # Plot the choropleth map
    fig = px.choropleth(
        data,
        locations="location",
        locationmode="country names", 
        color="new_cases_smoothed",
        hover_name="location",
        animation_frame=data['date'].dt.strftime('%Y-%m-%d'),
        title="COVID-19 Smoothed Cases Over Time",
        color_continuous_scale=px.colors.sequential.Plasma
    )

    fig.show()

