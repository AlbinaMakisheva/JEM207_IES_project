import matplotlib.pyplot as plt
import os

def visualize_stock(data, output_path='./visualizations/stock_trends.png'):
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.figure(figsize=(14, 8))
    
    # Plot price
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['Price'], label='Price', color='blue')
    plt.title("Pfizer Stock Price Trends")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.legend()

    # Plot volume
    plt.subplot(2, 1, 2)
    plt.bar(data.index, data['Volume'], label='Volume', color='red')
    plt.title("Pfizer Stock Trading Volume")
    plt.ylabel("Volume")
    plt.xlabel("Date")
    plt.grid(True)
    plt.legend()

    # Save and show 
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
