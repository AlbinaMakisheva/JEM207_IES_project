import pandas as pd
import plotly.express as px

def plot_cases_country(data):
    data['date'] = pd.to_datetime(data['date'])

    data['new_cases_smoothed'] = data.groupby('location')['new_cases_smoothed'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

    fig = px.choropleth(
        data,
        locations="location",  
        locationmode="country names",  
        color="new_cases_smoothed",  
        hover_name="location",  
        animation_frame="date",  
        title="COVID-19 Smoothed Cases Over Time",
        color_continuous_scale=px.colors.sequential.Plasma
    )

    fig.update_geos(showcoastlines=True, coastlinecolor="LightGray", projection_type="natural earth")
    fig.update_layout(
        sliders=[{
            'pad': {"t": 50},  
            'currentvalue': {"prefix": "Date: "}, 
        }]
    )
    fig.show()
