import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Define paths
INPUT_FILE = Path('data/processed/merged_data.csv')
OUTPUT_FILE = Path('data/processed/merged_data_with_dummy.csv')

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found at {INPUT_FILE}")

# Load the dataset
data = pd.read_csv(INPUT_FILE)


# Convert 'Date' to datetime format
data['date'] = pd.to_datetime(data['date'])

# Define key dates for major events
MAJOR_EVENT_DATES = {
    "WHO Declares Pandemic": "2020-03-11",
    "First Global Vaccination": "2020-12-08",
    "Vaccination Threshold Reached (85%)": "2021-07-30",
}

def create_dummy_variable(data, event_dates):
    # Convert event dates
    event_dates = {event: pd.to_datetime(date) for event, date in event_dates.items()}
    
    # Add a dummy variable column
    data['Dummy_Variable'] = data['date'].isin(event_dates.values()).astype(int)
    
    return data

data = create_dummy_variable(data, MAJOR_EVENT_DATES)

# Save the updated dataset
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
data.to_csv(OUTPUT_FILE, index=False)

# Remove duplicates and handle NaN values
data = data.drop_duplicates(subset=['date']).dropna(subset=['Close'])

def plot_stock_price(data, events):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))
    
    # Plot the stock price
    plt.plot(data['date'], data['Close'], label='Pfizer Stock Price', color='blue', linewidth=2)
    
    # Highlight major events
    for event, date in events.items():
        plt.axvline(pd.to_datetime(date), color='red', linestyle='--', alpha=0.7, label=event)
    
    # Set title and labels
    plt.title('Pfizer Stock Price with Major Events', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Closing Price (USD)', fontsize=14)
    plt.ylim(data['Close'].min() - 5, data['Close'].max() + 5) 
    plt.legend(loc='upper left', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    visualization_path = Path('visualizations/stock_price_with_events.png')
    visualization_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(visualization_path)
    plt.show()

plot_stock_price(data, MAJOR_EVENT_DATES)
