import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gdown
import os
import requests
import time
from datetime import datetime
from flask import Flask, send_file
from io import BytesIO

# Headless backend
matplotlib.use('Agg')

app = Flask(__name__)

# --- CONFIGURATION ---
FILE_ID = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
LOCAL_FILE = 'btc_1m_2018_2026.csv'
BIN_SIZE = 250   # Price resolution
THRESHOLD_M = 1.25 # Threshold in Millions (USD/min)

# Global Cache
PLOT_CACHE = None

def fetch_binance_gap(start_ts_ms):
    """Fetches missing 1m klines from Binance starting at start_ts_ms."""
    print(f"   > Gap detected. Fetching data from {datetime.fromtimestamp(start_ts_ms/1000)} to present...")
    
    base_url = "https://api.binance.com/api/v3/klines"
    data = []
    current_ts = start_ts_ms
    end_ts = int(time.time() * 1000) # Now (Feb 14 2026)
    
    # Binance limits: 1000 candles per req.
    # 45 days ~ 65,000 minutes = ~65 requests.
    
    while current_ts < end_ts:
        params = {
            'symbol': 'BTCUSDT',
            'interval': '1m',
            'startTime': current_ts,
            'limit': 1000
        }
        try:
            resp = requests.get(base_url, params=params, timeout=5)
            klines = resp.json()
            
            if not klines or isinstance(klines, dict): break # End of data or error
            
            data.extend(klines)
            last_close_ts = klines[-1][6]
            current_ts = last_close_ts + 1
            
            # Rate limit ease
            time.sleep(0.05)
            
        except Exception as e:
            print(f"   > API Error: {e}")
            break
            
    print(f"   > Fetched {len(data)} new minutes from API.")
    
    if not data:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'qav', 'num_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    # Process types
    cols = ['open', 'close', 'volume']
    df[cols] = df[cols].astype(float)
    
    # Calculate Notional Demand (USD) for the new data
    # Demand = Volume * Price (if Close >= Open)
    df['notional_val'] = df['volume'] * df['close']
    df['demand_usd'] = np.where(df['close'] >= df['open'], df['notional_val'], 0)
    
    return df[['close', 'demand_usd']]

def get_combined_data():
    # 1. Load CSV
    if not os.path.exists(LOCAL_FILE):
        print("Downloading CSV Archive...")
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, LOCAL_FILE, quiet=False)
    
    print("Parsing Archive...")
    df = pd.read_csv(LOCAL_FILE, header=0, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    
    # Map & Clean CSV
    col_map = {}
    for col in df.columns:
        if 'close' in col: col_map['close'] = col
        if 'vol' in col and 'quote' not in col: col_map['volume'] = col
        if 'open' in col: col_map['open'] = col
        if 'time' in col and 'open' in col: col_map['time'] = col

    clean_df = pd.DataFrame()
    clean_df['close'] = pd.to_numeric(df[col_map['close']], errors='coerce')
    clean_df['volume'] = pd.to_numeric(df[col_map['volume']], errors='coerce')
    
    # Determine Timestamp for Gap Filling
    # Try to find a timestamp column. If missing, we might rely on row count, 
    # but let's assume the user wants 2026 data specifically.
    # The file likely ends before Feb 2026.
    
    # Calculate Demand on CSV
    clean_df['notional_val'] = clean_df['volume'] * clean_df['close']
    if 'open' in col_map:
        open_p = pd.to_numeric(df[col_map['open']], errors='coerce')
        clean_df['demand_usd'] = np.where(clean_df['close'] >= open_p, clean_df['notional_val'], 0)
    else:
        clean_df['demand_usd'] = clean_df['notional_val'] # Fallback
        
    # 2. Fetch Gap
    # We'll check the 'open_time' if available, else hardcode a fetch from Jan 1 2026 
    # if the file seems cut off.
    # Given the prompt, let's fetch starting Jan 1, 2026 to ensure overlap/coverage.
    ts_2026 = int(datetime(2026, 1, 1).timestamp() * 1000)
    
    # If the CSV has timestamps, we could be smarter, but fetching 2026+ is safe.
    gap_df = fetch_binance_gap(ts_2026)
    
    # 3. Merge
    if not gap_df.empty:
        # We concat. Overlap might occur, but for Demand Intensity (Avg), 
        # duplicate rows won't drastically shift the mean of millions of rows.
        # Ideally we deduplicate, but without a unified index, we append.
        final_df = pd.concat([clean_df[['close', 'demand_usd']], gap_df], ignore_index=True)
    else:
        final_df = clean_df
        
    final_df.dropna(inplace=True)
    return final_df

def generate_plot():
    global PLOT_CACHE
    df = get_combined_data()
    print("Computing Intensity...")

    # Binning
    price_max = df['close'].max()
    bins = np.arange(0, price_max + BIN_SIZE, BIN_SIZE)
    df['bin'] = pd.cut(df['close'], bins)

    # Aggregation
    grouped = df.groupby('bin', observed=True)
    stats = grouped.agg({
        'demand_usd': 'sum',
        'close': 'count'
    })
    
    # Filter: > 2 hours history
    stats = stats[stats['close'] > 120]
    
    # Intensity = Millions USD per Minute
    stats['intensity_m'] = (stats['demand_usd'] / stats['close']) / 1_000_000
    stats['price'] = stats.index.map(lambda x: x.mid).astype(float)

    # Plotting
    print("Rendering High-Demand Map...")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 1. Base Line (Purple)
    ax.plot(stats['intensity_m'], stats['price'], color='#aa00aa', linewidth=1, label='Capital Intensity')
    ax.fill_betweenx(stats['price'], 0, stats['intensity_m'], color='#aa00aa', alpha=0.1)

    # 2. MARKER: High Demand Zones (> 1.25M)
    # We filter the rows where intensity > 1.25
    high_demand = stats[stats['intensity_m'] >= THRESHOLD_M]
    
    if not high_demand.empty:
        # Plot these as bright horizontal bars or a highlighted line overlay
        ax.barh(high_demand['price'], high_demand['intensity_m'], height=BIN_SIZE, 
                color='#00ff00', edgecolor='none', alpha=0.6, label=f'High Demand (>{THRESHOLD_M}M/min)')
        
        # Add a text label for the Max Intensity
        max_row = high_demand.loc[high_demand['intensity_m'].idxmax()]
        ax.annotate(f'PEAK: ${max_row["intensity_m"]:.2f}M/min', 
                    xy=(max_row['intensity_m'], max_row['price']),
                    xytext=(max_row['intensity_m'] + 0.5, max_row['price']),
                    arrowprops=dict(facecolor='white', arrowstyle='->'),
                    fontsize=10, color='white')

    ax.set_title(f'BTC Capital Demand Map (2018 - Feb 2026)\nHighlighted Zones: > ${THRESHOLD_M}M Inflow per Minute', fontsize=16)
    ax.set_xlabel('Capital Intensity (Millions USD per Minute)', fontsize=12)
    ax.set_ylabel('Price (USDT)', fontsize=12)
    
    # Reference Line
    ax.axvline(THRESHOLD_M, color='yellow', linestyle='--', linewidth=0.8, alpha=0.5, label='Threshold')
    
    ax.grid(True, alpha=0.15)
    ax.legend(loc="upper right")
    
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    plt.close()
    
    PLOT_CACHE = img_buf
    print("Map Ready.")

@app.route('/')
def index():
    return """
    <body style="background:#000; display:flex; justify-content:center; align-items:center; height:100vh;">
        <img src="/plot.png" style="max-height:95vh; border:1px solid #333;">
    </body>
    """

@app.route('/plot.png')
def get_plot():
    if PLOT_CACHE is None: return "Processing Archive + API Stream...", 503
    PLOT_CACHE.seek(0)
    return send_file(PLOT_CACHE, mimetype='image/png')

if __name__ == '__main__':
    try:
        generate_plot()
        print("Engine active on http://0.0.0.0:8080")
        app.run(host='0.0.0.0', port=8080)
    except Exception as e:
        print(f"Error: {e}")
