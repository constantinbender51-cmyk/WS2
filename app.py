import io
import requests
import pandas as pd
import matplotlib
# Use the 'Agg' backend so matplotlib doesn't try to open a GUI window on a server
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from flask import Flask, send_file

app = Flask(__name__)

def fetch_binance_data():
    """Fetches the last 30 days of 1h BTCUSDT data from Binance."""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "limit": 720  # 30 days * 24 hours = 720 hours
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    # Define columns based on Binance API documentation
    cols = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 
            'CloseTime', 'QuoteVolume', 'NumTrades', 
            'TakerBuyBaseVol', 'TakerBuyQuoteVol', 'Ignore']
    
    df = pd.DataFrame(data, columns=cols)
    
    # Convert necessary columns to float
    df['Open'] = df['Open'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    
    return df

def process_movement_data(df):
    """Processes directional movement and scales volume for bubbles."""
    # Green body -> +1 (forward), Red body -> -1 (backward)
    df['Direction'] = df.apply(lambda row: 1 if row['Close'] >= row['Open'] else -1, axis=1)
    
    # Cumulative movement for the X-axis
    df['Movement_X'] = df['Direction'].cumsum()
    
    # Color map for the bubbles
    df['Color'] = df['Direction'].map({1: '#26a69a', -1: '#ef5350'}) # Standard crypto green/red
    
    # Normalize volume to use as bubble sizes (min size 20, max size 600)
    min_vol = df['Volume'].min()
    max_vol = df['Volume'].max()
    df['BubbleSize'] = 20 + 580 * ((df['Volume'] - min_vol) / (max_vol - min_vol))
    
    return df

@app.route('/')
def serve_plot():
    # 1. Fetch & Process Data
    df = fetch_binance_data()
    df = process_movement_data(df)
    
    # 2. Create Plot
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='#121212')
    ax.set_facecolor('#121212')
    
    # Faint line to show the chronological path
    ax.plot(df['Movement_X'], df['Close'], color='white', alpha=0.15, linewidth=1, zorder=1)
    
    # Scatter plot for Momentum Bubbles
    ax.scatter(
        df['Movement_X'], 
        df['Close'], 
        s=df['BubbleSize'], 
        c=df['Color'], 
        alpha=0.7, 
        edgecolors='black',
        linewidth=0.5,
        zorder=2
    )
    
    # Styling
    ax.set_title("BTCUSDT Momentum Bubbles (Last 30 Days, 1h Interval)", color='white', fontsize=16)
    ax.set_xlabel("Cumulative Movement (Green = Forward +1, Red = Backward -1)", color='white', fontsize=12)
    ax.set_ylabel("Price (USDT)", color='white', fontsize=12)
    
    ax.tick_params(colors='white')
    ax.grid(True, linestyle='--', alpha=0.2, color='white')
    
    # 3. Save to memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    
    # 4. Serve the image
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    # Serve on 0.0.0.0 so it is exposed to your local network / public IP
    print("Serving Momentum Bubbles on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)