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
        print("Downloading dataset...")
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, OUTPUT_FILE, quiet=False)
    else:
        print("File already exists. Skipping download.")

def process_intensity():
    print("Loading dataset...")
    
    # FIX: 'header=0' tells pandas the first row contains column names like "close", "open"
    # 'low_memory=False' prevents warning on mixed types if parsing is tricky
    df = pd.read_csv(OUTPUT_FILE, header=0, low_memory=False)
    
    # 1. Standardize Column Names (Strip spaces, lowercase)
    df.columns = df.columns.str.strip().str.lower()
    print(f"Columns found: {list(df.columns)}")

    # 2. Map Columns to Standard Names
    # We look for standard keywords in the file's header
    col_map = {}
    for col in df.columns:
        if 'close' in col: col_map['close'] = col
        elif 'open' in col: col_map['open'] = col
        elif 'high' in col: col_map['high'] = col
        elif 'low' in col: col_map['low'] = col
        elif 'volume' in col or 'vol' in col: col_map['volume'] = col
        elif 'taker_buy' in col: col_map['taker_buy'] = col

    # Check if we have the critical 'close' and 'volume' columns
    if 'close' not in col_map or 'volume' not in col_map:
        raise ValueError(f"Could not identify 'close' or 'volume' columns in: {df.columns}")
    
    # 3. Create a clean DataFrame with mapped columns
    clean_df = pd.DataFrame()
    clean_df['close'] = df[col_map['close']]
    clean_df['volume'] = df[col_map['volume']]
    
    # If 'open' exists, use it for Directional Logic
    if 'open' in col_map:
        clean_df['open'] = df[col_map['open']]
    else:
        # Fallback: Use prev close (less accurate but works)
        clean_df['open'] = clean_df['close'].shift(1)

    # 4. Handle "Willingness to Buy"
    # If explicit 'taker_buy' column exists, use it. Otherwise, approximate.
    if 'taker_buy' in col_map:
        print("Using explicit Taker Buy Volume.")
        clean_df['demand_vol'] = df[col_map['taker_buy']]
    else:
        print("Using Directional Approximation (Close > Open).")
        # If Price Up -> Demand (Volume), If Price Down -> Supply (0 Demand)
        clean_df['demand_vol'] = np.where(clean_df['close'] >= clean_df['open'], clean_df['volume'], 0)

    # Ensure numeric types (coerce errors to NaN then drop)
    clean_df['close'] = pd.to_numeric(clean_df['close'], errors='coerce')
    clean_df['demand_vol'] = pd.to_numeric(clean_df['demand_vol'], errors='coerce')
    clean_df.dropna(subset=['close', 'demand_vol'], inplace=True)

    # --- BINNING & INTENSITY ---
    print("Calculating Demand Intensity...")
    
    # Create Price Bins
    price_max = clean_df['close'].max()
    bins = np.arange(0, price_max + BIN_SIZE, BIN_SIZE)
    clean_df['price_bin'] = pd.cut(clean_df['close'], bins)
    
    # Aggregate
    grouped = clean_df.groupby('price_bin', observed=True)
    analysis = grouped.agg({
        'demand_vol': 'sum',      # Total BTC Bought
        'close': 'count'          # Minutes Spent
    }).rename(columns={'close': 'minutes_spent'})
    
    # Filter noise (Must have spent at least 2 hours at this price level history-wide)
    analysis = analysis[analysis['minutes_spent'] > 120]
    
    # Intensity = Volume / Time
    analysis['buy_intensity'] = analysis['demand_vol'] / analysis['minutes_spent']
    analysis['price_mid'] = analysis.index.map(lambda x: x.mid).astype(float)
    
    return analysis

def plot_demand_intensity(df):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plotting Line
    y = df['price_mid']
    x = df['buy_intensity']
    
    ax.plot(x, y, color='#00ffcc', linewidth=2, label='Demand Intensity')
    ax.fill_betweenx(y, 0, x, color='#00ffcc', alpha=0.2)
    
    ax.set_title('BTC Demand Intensity (2018-2026)\n(Average Buying Volume per Minute at Price Level)', fontsize=14)
    ax.set_ylabel('Price (USDT)')
    ax.set_xlabel('Intensity (BTC Bought / Minute)')
    ax.legend()
    ax.grid(True, alpha=0.15)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    download_data()
    try:
        data = process_intensity()
        plot_demand_intensity(data)
    except Exception as e:
        print(f"Error: {e}")
