import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gdown
import os
from flask import Flask, send_file
from io import BytesIO

# Non-interactive backend for server environments
matplotlib.use('Agg')

app = Flask(__name__)

# --- CONFIGURATION ---
FILE_ID = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
Local_FILE = 'btc_1m_2018_2026.csv'
BIN_SIZE = 250  # Resolution of the demand curve ($250 bins)

# Global Buffer for the Plot
PLOT_CACHE = None

def download_and_load_data():
    """Fetches CSV and returns a cleaned DataFrame."""
    if not os.path.exists(Local_FILE):
        print("Downloading dataset from Drive...")
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, Local_FILE, quiet=False)
    
    print("Parsing CSV...")
    # header=0 handles the "close", "open" text row
    df = pd.read_csv(Local_FILE, header=0, low_memory=False)
    
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Map essential columns
    col_map = {}
    for col in df.columns:
        if 'close' in col: col_map['close'] = col
        if 'vol' in col and 'quote' not in col: col_map['volume'] = col
        if 'open' in col: col_map['open'] = col

    if 'close' not in col_map or 'volume' not in col_map:
        raise ValueError(f"Required columns missing. Found: {df.columns}")

    # Extract relevant data
    clean_df = pd.DataFrame()
    clean_df['close'] = pd.to_numeric(df[col_map['close']], errors='coerce')
    clean_df['volume'] = pd.to_numeric(df[col_map['volume']], errors='coerce')
    
    # Directional Approximation for Demand
    # If Open exists, use Close > Open. Else assume all volume is activity.
    if 'open' in col_map:
        open_price = pd.to_numeric(df[col_map['open']], errors='coerce')
        # Demand = Volume where Price Closed UP
        clean_df['demand_vol'] = np.where(clean_df['close'] >= open_price, clean_df['volume'], 0)
    else:
        clean_df['demand_vol'] = clean_df['volume'] # Fallback

    clean_df.dropna(inplace=True)
    return clean_df

def generate_plot():
    """Computes logic and renders image to buffer."""
    global PLOT_CACHE
    
    df = download_and_load_data()
    print("Computing Demand Intensity...")

    # 1. Binning
    price_max = df['close'].max()
    bins = np.arange(0, price_max + BIN_SIZE, BIN_SIZE)
    df['bin'] = pd.cut(df['close'], bins)

    # 2. Aggregation (Volume vs Time)
    grouped = df.groupby('bin', observed=True)
    stats = grouped.agg({
        'demand_vol': 'sum',      # Total "Willingness" (Gross)
        'close': 'count'          # Total Time (Minutes)
    })
    
    # 3. Intensity Calculation
    # Filter: Must have spent > 2 hours total at this price to be statistically significant
    stats = stats[stats['close'] > 120]
    
    # Intensity = Volume / Minutes
    # "How hard did they buy per minute of opportunity?"
    stats['intensity'] = stats['demand_vol'] / stats['close']
    stats['price'] = stats.index.map(lambda x: x.mid).astype(float)

    # 4. Plotting
    print("Rendering...")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # X = Intensity (Buying Power), Y = Price
    ax.plot(stats['intensity'], stats['price'], color='#00ffcc', linewidth=1.5)
    ax.fill_betweenx(stats['price'], 0, stats['intensity'], color='#00ffcc', alpha=0.2)
    
    ax.set_title('BTC Demand Schedule (2018-2026)\nNormalized for Time Duration (Volume / Minute)', fontsize=16)
    ax.set_xlabel('Demand Intensity (Avg BTC Bought per Minute)', fontsize=12)
    ax.set_ylabel('Price (USDT)', fontsize=12)
    ax.grid(True, alpha=0.15)
    
    # Save to buffer
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    plt.close()
    
    PLOT_CACHE = img_buf
    print("Plot ready.")

@app.route('/')
def index():
    return """
    <body style="background:#111; margin:0; display:flex; justify-content:center; align-items:center; height:100vh;">
        <img src="/plot.png" style="max-height:95vh; border:1px solid #333; box-shadow:0 0 20px #00ffcc22;">
    </body>
    """

@app.route('/plot.png')
def get_plot():
    if PLOT_CACHE is None:
        return "Initializing...", 503
    PLOT_CACHE.seek(0)
    return send_file(PLOT_CACHE, mimetype='image/png')

if __name__ == '__main__':
    # Pre-compute on startup to avoid delay on request
    try:
        generate_plot()
        print("Serving on http://0.0.0.0:8080")
        app.run(host='0.0.0.0', port=8080)
    except Exception as e:
        print(f"Fatal Error: {e}")
