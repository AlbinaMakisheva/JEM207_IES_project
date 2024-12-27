import pandas as pd
import matplotlib.pyplot as plt

def plot_cases_vs_stock(data):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(data['date'], data['new_cases_smoothed'], color='orange', label='COVID-19 Cases')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('COVID-19 Cases', color='orange', fontsize=12)
    ax2 = ax1.twinx()
    ax2.plot(data['date'], data['Close'], color='blue', label='Pfizer Stock Price')
    ax2.set_ylabel('Pfizer Stock Price (USD)', color='blue', fontsize=12)
    fig.suptitle('COVID-19 Cases vs Pfizer Stock Price', fontsize=14)
    fig.tight_layout()
    plt.grid()
    plt.show()

