import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gdown
import os
from flask import Flask, send_file
from io import BytesIO

# Non-interactive backend
matplotlib.use('Agg')

app = Flask(__name__)

# --- CONFIGURATION ---
FILE_ID = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
Local_FILE = 'btc_1m_2018_2026.csv'
BIN_SIZE = 500  # $500 Price Buckets

PLOT_CACHE = None

def download_and_load_data():
    if not os.path.exists(Local_FILE):
        print("Downloading dataset...")
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, Local_FILE, quiet=False)
    
    print("Parsing CSV...")
    df = pd.read_csv(Local_FILE, header=0, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    
    # Map columns
    col_map = {}
    for col in df.columns:
        if 'close' in col: col_map['close'] = col
        if 'vol' in col and 'quote' not in col: col_map['volume'] = col
        if 'open' in col: col_map['open'] = col

    # Extract Data
    clean_df = pd.DataFrame()
    clean_df['close'] = pd.to_numeric(df[col_map['close']], errors='coerce')
    clean_df['volume'] = pd.to_numeric(df[col_map['volume']], errors='coerce')
    
    # --- CRITICAL FIX: CONVERT TO USD (NOTIONAL VALUE) ---
    # We calculate "Money Flow" instead of "Coin Flow"
    # This neutralizes the "Low Price = High Quantity" bias.
    clean_df['notional_val'] = clean_df['volume'] * clean_df['close']
    
    # Directional Logic (Estimate Demand Side)
    if 'open' in col_map:
        open_price = pd.to_numeric(df[col_map['open']], errors='coerce')
        # If Price Up -> The Money Flow was Demand
        clean_df['demand_usd'] = np.where(clean_df['close'] >= open_price, clean_df['notional_val'], 0)
    else:
        # Fallback
        clean_df['demand_usd'] = clean_df['notional_val']

    clean_df.dropna(inplace=True)
    return clean_df

def generate_plot():
    global PLOT_CACHE
    df = download_and_load_data()
    print("Computing USD Demand Intensity...")

    # 1. Binning
    price_max = df['close'].max()
    bins = np.arange(0, price_max + BIN_SIZE, BIN_SIZE)
    df['bin'] = pd.cut(df['close'], bins)

    # 2. Aggregation (Money vs Time)
    grouped = df.groupby('bin', observed=True)
    stats = grouped.agg({
        'demand_usd': 'sum',      # Total Dollars Spent buying
        'close': 'count'          # Total Minutes Spent
    })
    
    # Filter statistical noise (must have > 2 hours history)
    stats = stats[stats['close'] > 120]
    
    # 3. Intensity Calculation
    # Result: "Average Millions of Dollars Inflow per Minute"
    stats['intensity'] = stats['demand_usd'] / stats['close']
    stats['price'] = stats.index.map(lambda x: x.mid).astype(float)

    # 4. Plotting
    print("Rendering...")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # X = Intensity (USD), Y = Price
    ax.plot(stats['intensity'], stats['price'], color='#ff00ff', linewidth=1.5)
    ax.fill_betweenx(stats['price'], 0, stats['intensity'], color='#ff00ff', alpha=0.2)
    
    ax.set_title('BTC Capital Demand Schedule (USD)\n"Willingness to Spend Money"', fontsize=16)
    ax.set_xlabel('Capital Intensity (Avg USD Inflow per Minute)', fontsize=12)
    ax.set_ylabel('Price (USDT)', fontsize=12)
    
    # Format X axis to Millions
    def millions(x, pos):
        return '$%1.1fM' % (x * 1e-6)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(millions))
    
    ax.grid(True, alpha=0.15)
    
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    plt.close()
    
    PLOT_CACHE = img_buf
    print("Plot ready.")

@app.route('/')
def index():
    return """
    <body style="background:#000; display:flex; justify-content:center; align-items:center; height:100vh;">
        <img src="/plot.png" style="max-height:95vh; border:1px solid #333;">
    </body>
    """

@app.route('/plot.png')
def get_plot():
    if PLOT_CACHE is None: return "Processing...", 503
    PLOT_CACHE.seek(0)
    return send_file(PLOT_CACHE, mimetype='image/png')

if __name__ == '__main__':
    try:
        generate_plot()
        print("Serving on http://0.0.0.0:8080")
        app.run(host='0.0.0.0', port=8080)
    except Exception as e:
        print(f"Error: {e}")
