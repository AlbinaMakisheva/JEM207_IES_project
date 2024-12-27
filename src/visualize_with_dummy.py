import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
print("Current Working Directory:", os.getcwd())


# Load the processed dataset
data = pd.read_csv(r'C:\Users\campo\OneDrive\Área de Trabalho\Erasmus - Karlova\Python\JEM207_IES_project\data\processed\merged_data.csv')


# Convert 'Date' to datetime format
data['date'] = pd.to_datetime(data['date']) 

# Define key dates for major events
major_event_dates = {
        "WHO Declares Pandemic": "2020-03-11",
        "First Global Vaccination": "2020-12-08",
        "Vaccination threshold reached (85%)": "2021-07-30"    #as of July 2021, almost 85% of vaccines have been administered in high- and upper-middle-income countries, and over 75% have been administered in only 10 countries alone. 
    }

# Create a Dummy Variable for Major Events
def create_dummy_variable(data):
    """
    Create dummy variables for major events affecting Pfizer stock performance.
    """
    

    # Create a new column initialized to 0
    data['Dummy_Variable'] = 0

    # Set Dummy_Variable to 1 for rows matching the major events
    for event, date in major_event_dates.items():
        data.loc[data['date'] == pd.to_datetime(date), 'Dummy_Variable'] = 1

    return data

# Apply the function to create the dummy variable
data = create_dummy_variable(data)

# Save the updated dataset 
file_path = r'C:\Users\campo\OneDrive\Área de Trabalho\Erasmus - Karlova\Python\JEM207_IES_project\data\processed\merged_data_with_dummy.csv'
data.to_csv(file_path, index=False)

# Remove duplicates and handle NaN values
data = data.drop_duplicates(subset=['date']).dropna(subset=['Close'])

# Pfizer Stock Price with Major Events Highlighted

def plot_stock_price():
    """
    Plot Pfizer's stock price with major events highlighted.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot the stock price
    plt.plot(data['date'], data['Close'], label='Pfizer Stock Price', color='blue')  # Adjusted column name
    
    # Highlight major events
    events = [
        ("WHO Declares Pandemic", "2020-03-11"),
        ("First Global Vaccination", "2020-12-08"),
        ("Vaccination threshold reached (85%)", "2021-07-30"),
    ]
    for event, date in events:
        plt.axvline(pd.to_datetime(date), color='red', linestyle='--', alpha=0.7, label=event)
    
    # Set title and labels
    plt.title('Pfizer Stock Price with Major Events', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Closing Price (USD)', fontsize=12)
    
    # Adjust y-axis limits to zoom out
    plt.ylim(data['Close'].min() - 5, data['Close'].max() + 5)
    
    # Add legend and grid
    plt.legend()
    plt.grid()
    plt.show()

