import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gdown
import os

# --- CONFIGURATION ---
FILE_ID = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
OUTPUT_FILE = 'btc_1m_2018_2026.csv'
BIN_SIZE = 500  # Price bin width (e.g., every $500)

def download_data():
    if not os.path.exists(OUTPUT_FILE):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, OUTPUT_FILE, quiet=False)
    else:
        print("File already exists. Skipping download.")

def process_intensity():
    # Load Data (Assuming standard Binance columns without header)
    # If header exists, change 'header=None' to 'header=0'
    print("Loading dataset...")
    df = pd.read_csv(OUTPUT_FILE, header=None)
    
    # Binance 1m CSV Format:
    # 0: Open Time, 1: Open, 2: High, 3: Low, 4: Close, 5: Vol, 
    # 6: Close Time, 7: Quote Vol, 8: Trades, 9: Taker Buy Base, ...
    df.columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'qav', 'num_trades', 'taker_buy_base', 
        'taker_buy_quote', 'ignore'
    ]
    
    # Optimize types
    cols = ['open', 'high', 'low', 'close', 'taker_buy_base']
    df[cols] = df[cols].astype(float)
    
    # 1. Price Binning
    # We use the 'Close' price to determine which bin the minute belongs to.
    # A more precise method uses (High+Low)/2, but Close is sufficient for 4M rows.
    price_min = df['low'].min()
    price_max = df['high'].max()
    bins = np.arange(0, price_max + BIN_SIZE, BIN_SIZE) # Start at 0 to catch early days
    
    df['price_bin'] = pd.cut(df['close'], bins)
    
    # 2. Aggregation (The "Willingness" Calculation)
    # Group by Price Bin
    print("Aggregating volume by price bin...")
    grouped = df.groupby('price_bin', observed=True)
    
    # Calculate Metrics:
    # A. Total Taker Buy Volume (Raw Willingness)
    # B. Count of Minutes (Time Spent)
    analysis = grouped.agg({
        'taker_buy_base': 'sum',  # Total Buy Vol
        'close': 'count'          # Total Minutes (Time)
    }).rename(columns={'close': 'minutes_spent'})
    
    # 3. Time Normalization (Intensity)
    # Intensity = Volume / Time
    # "How many BTC were bought per minute while price was in this bin?"
    analysis['buy_intensity'] = analysis['taker_buy_base'] / analysis['minutes_spent']
    
    # Get the mid-point of each bin for plotting
    analysis['price_mid'] = analysis.index.map(lambda x: x.mid).astype(float)
    
    # Filter out empty bins or bins with very low time (outliers)
    analysis = analysis[analysis['minutes_spent'] > 60] # Must have spent at least 1 hour total
    
    return analysis

def plot_demand_intensity(df):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Y-Axis = Price
    # X-Axis = Intensity (BTC Bought / Minute)
    y = df['price_mid']
    x = df['buy_intensity']
    
    # Plot as a horizontal bar chart or a line
    # A line is better to see the "Structure" of demand
    ax.plot(x, y, color='#00ffcc', linewidth=2, label='Demand Intensity (Buy Vol / Time)')
    
    # Fill for visual weight
    ax.fill_betweenx(y, 0, x, color='#00ffcc', alpha=0.2)
    
    ax.set_title('BTC Time-Normalized Demand Schedule (2018-2026)\n"True Willingness to Buy"', fontsize=14)
    ax.set_ylabel('Price (USDT)')
    ax.set_xlabel('Intensity: Average Aggressor Buy Volume per Minute (BTC/min)')
    
    # Log scale on X might be necessary if early years had massive volume
    # ax.set_xscale('log') 
    
    ax.grid(True, alpha=0.2)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    download_data()
    data = process_intensity()
    plot_demand_intensity(data)
