import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gdown
import os

# --- CONFIGURATION ---
FILE_ID = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
OUTPUT_FILE = 'btc_1m_2018_2026.csv'
BIN_SIZE = 250  # Price bin width ($250)

def download_data():
    if not os.path.exists(OUTPUT_FILE):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, OUTPUT_FILE, quiet=False)
    else:
        print("File already exists. Skipping download.")

def process_intensity():
    print("Loading dataset...")
    # Read without header first to detect shape
    df = pd.read_csv(OUTPUT_FILE, header=None)
    
    # --- COLUMN DETECTION & NAMING ---
    if df.shape[1] == 6:
        print("Detected 6-column format (OHLCV). Using Directional Approximation.")
        df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        
        # APPROXIMATION: If Close > Open, treat full volume as Demand.
        # This is a heuristic for OHLC data lacking tick direction.
        df['taker_buy_base'] = np.where(df['close'] > df['open'], df['volume'], 0)
        
    elif df.shape[1] >= 12:
        print("Detected 12-column format (Binance Standard). Using Explicit Buy Volume.")
        df.columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'qav', 'num_trades', 'taker_buy_base', 
            'taker_buy_quote', 'ignore'
        ]
    else:
        # Fallback for unexpected formats (e.g., 9 cols)
        print(f"Detected {df.shape[1]} columns. Attempting to map first 6.")
        df = df.iloc[:, :6]
        df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        df['taker_buy_base'] = np.where(df['close'] > df['open'], df['volume'], 0)

    # Type Conversion
    cols = ['close', 'taker_buy_base']
    df[cols] = df[cols].astype(float)
    
    # --- BINNING LOGIC ---
    print("Binning prices...")
    price_max = df['close'].max()
    bins = np.arange(0, price_max + BIN_SIZE, BIN_SIZE)
    
    df['price_bin'] = pd.cut(df['close'], bins)
    
    # --- INTENSITY CALCULATION ---
    # 1. Sum Total Buy Volume per Bin (Total Willingness)
    # 2. Count Total Minutes per Bin (Time Spent)
    print("Calculating intensity...")
    grouped = df.groupby('price_bin', observed=True)
    
    analysis = grouped.agg({
        'taker_buy_base': 'sum', 
        'close': 'count'
    }).rename(columns={'close': 'minutes_spent'})
    
    # Filter noise: Must have spent at least 2 hours (120 mins) in that price bucket
    analysis = analysis[analysis['minutes_spent'] > 120]
    
    # Intensity = Volume / Time
    analysis['buy_intensity'] = analysis['taker_buy_base'] / analysis['minutes_spent']
    
    # Midpoint for Plotting
    analysis['price_mid'] = analysis.index.map(lambda x: x.mid).astype(float)
    
    return analysis

def plot_demand_intensity(df):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 12))
    
    y = df['price_mid']
    x = df['buy_intensity']
    
    # Plot Logic
    ax.plot(x, y, color='#00ffcc', linewidth=1.5, label='Demand Intensity')
    ax.fill_betweenx(y, 0, x, color='#00ffcc', alpha=0.2)
    
    # Formatting
    ax.set_title('BTC Historical Demand Intensity (2018-2026)\n(Avg Buy Volume per Minute at Price Level)', fontsize=14)
    ax.set_ylabel('Price (USDT)')
    ax.set_xlabel('Intensity (BTC Bought / Minute)')
    ax.legend()
    ax.grid(True, alpha=0.15)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    download_data()
    data = process_intensity()
    plot_demand_intensity(data)
