import requests
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, send_file, render_template_string
from io import BytesIO
import time
from datetime import datetime
import threading

# Headless backend
matplotlib.use('Agg')

app = Flask(__name__)

# --- CONFIGURATION ---
SYMBOL = 'BTCUSDT'
INTERVAL = '1m'
START_DATE = '2025-11-01'
END_DATE = '2026-02-14'
BIN_COUNT = 250 

# Global Memory Buffer
IMG_CACHE = None

def get_timestamp(date_str):
    return int(datetime.strptime(date_str, '%Y-%m-%d').timestamp() * 1000)

def fetch_market_data():
    print(f"[{datetime.now()}] ACQUIRING VECTOR STREAM: {SYMBOL} ({START_DATE} -> {END_DATE})...")
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = get_timestamp(START_DATE)
    end_ts = get_timestamp(END_DATE)
    
    data = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        params = {
            'symbol': SYMBOL, 'interval': INTERVAL,
            'startTime': current_ts, 'endTime': end_ts, 'limit': 1000
        }
        
        try:
            resp = requests.get(base_url, params=params, timeout=10)
            resp.raise_for_status()
            klines = resp.json()
            
            if not klines: break
                
            data.extend(klines)
            current_ts = klines[-1][6] + 1
            time.sleep(0.05) # Aggressive rate limit
            print(f"\rVectors loaded: {len(data)}", end="")
            
        except Exception as e:
            print(f"\nStream interrupt: {e}")
            break
            
    print("\nVector acquisition complete.")
    
    df = pd.DataFrame(data, columns=[
        'opentime', 'open', 'high', 'low', 'close', 'volume', 
        'closetime', 'qav', 'num_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    cols = ['open', 'high', 'low', 'close', 'volume']
    df[cols] = df[cols].astype(float)
    return df

def compute_and_render():
    global IMG_CACHE
    df = fetch_market_data()
    
    if df.empty:
        print("Dataset empty. Aborting render.")
        return

    # Binning Logic
    price_min = df['low'].min()
    price_max = df['high'].max()
    bins = np.linspace(price_min, price_max, BIN_COUNT + 1)
    
    # Vectorized Classification
    df['mid'] = (df['high'] + df['low']) / 2
    # 1 = Buy (Demand), -1 = Sell (Supply)
    df['direction'] = np.where(df['close'] >= df['open'], 1, -1)
    df['bin_idx'] = np.digitize(df['mid'], bins) - 1
    df['bin_idx'] = df['bin_idx'].clip(0, BIN_COUNT - 1)
    
    vol_buy = df[df['direction'] == 1].groupby('bin_idx')['volume'].sum()
    vol_sell = df[df['direction'] == -1].groupby('bin_idx')['volume'].sum()
    
    all_indices = pd.Index(range(BIN_COUNT))
    vol_buy = vol_buy.reindex(all_indices, fill_value=0).values
    vol_sell = vol_sell.reindex(all_indices, fill_value=0).values
    
    # Plotting
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 9))
    
    y_pos = bins[:-1]
    height = bins[1] - bins[0]
    
    ax.barh(y_pos, vol_buy, height=height, color='#00ff00', alpha=0.6, label='Demand', align='edge')
    ax.barh(y_pos, -vol_sell, height=height, color='#ff0000', alpha=0.6, label='Supply', align='edge')
    
    ax.set_title(f'BTC/USDT Liquidity Profile | {START_DATE} :: {END_DATE}')
    ax.set_xlabel('Volume < Supply | Demand >')
    ax.axvline(0, color='white', linewidth=0.5)
    ax.legend()
    
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plt.close()
    
    IMG_CACHE = img
    print("Render complete. Cache hot.")

@app.route('/')
def index():
    return """
    <body style="background:#000; display:flex; justify-content:center; margin:0;">
        <img src="/plot.png" style="height:100vh; border:1px solid #333;">
    </body>
    """

@app.route('/plot.png')
def plot():
    if IMG_CACHE:
        IMG_CACHE.seek(0)
        return send_file(IMG_CACHE, mimetype='image/png')
    return "Initializing...", 503

if __name__ == '__main__':
    # Blocking Fetch on Startup
    compute_and_render()
    
    print("Serving on port 8080...")
    app.run(host='0.0.0.0', port=8080)
