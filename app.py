import io
import requests
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from flask import Flask, send_file

app = Flask(__name__)

# ==========================================
# TIME SENSITIVITY MULTIPLIER
# ==========================================
# Because 1h crypto moves are usually small fractions (e.g., 0.02 for 2%),
# (close-open)/open rarely exceeds -1.0 to actually force time backwards.
# This multiplier scales the price action so time literally regresses on red candles,
# forming the enclosed "Momentum Bubbles".
SENSITIVITY_MULTIPLIER = 100  
# ==========================================

def fetch_binance_data():
    """Fetches the last 30 days of 1h BTCUSDT data from Binance (720 candles)."""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "limit": 720
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    cols = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 
            'CloseTime', 'QuoteVolume', 'NumTrades', 
            'TakerBuyBaseVol', 'TakerBuyQuoteVol', 'Ignore']
    
    df = pd.DataFrame(data, columns=cols)
    df['Open'] = df['Open'].astype(float)
    df['Close'] = df['Close'].astype(float)
    
    return df

def process_relativistic_time(df):
    """
    Calculates time drift using the user's formula:
    time_step = 1 + (close - open) / open
    """
    # Calculate the fractional movement (e.g., 0.05 for a 5% gain)
    fractional_move = (df['Close'] - df['Open']) / df['Open']
    
    # Apply the formula with the multiplier to allow for negative time steps
    df['Time_Step'] = 1 + (fractional_move * SENSITIVITY_MULTIPLIER)
    
    # X-axis is the cumulative sum of these warped time steps
    df['X_Movement'] = df['Time_Step'].cumsum()
    
    return df

@app.route('/')
def serve_plot():
    # 1. Fetch & Process Data
    df = fetch_binance_data()
    df = process_relativistic_time(df)
    
    # 2. Create Plot
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    
    # Plot the continuous trajectory of the points
    ax.plot(
        df['X_Movement'], 
        df['Close'], 
        color='#00ffcc',      # Neon cyan line
        linewidth=0.8,        # Thin line to trace the orbits/bubbles
        alpha=0.7,
        marker='o',           # Draw the literal points
        markersize=3,         
        markerfacecolor='white',
        markeredgecolor='none',
        zorder=2
    )
    
    # 3. Styling
    ax.set_title("BTCUSDT Relativistic Momentum Bubbles", color='white', fontsize=16, pad=20)
    
    formula_text = f"X-Axis Formula: Time += 1 + [ (Close - Open) / Open * {SENSITIVITY_MULTIPLIER} ]"
    ax.set_xlabel(f"Warped Time Drift\n{formula_text}", color='white', fontsize=11)
    ax.set_ylabel("Price (USDT)", color='white', fontsize=12)
    
    ax.tick_params(colors='white')
    ax.grid(True, linestyle='-', alpha=0.1, color='white')
    
    # Remove borders for a cleaner aesthetic
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    
    # 4. Save to memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    
    # 5. Serve the image
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    print("Serving Momentum Bubbles on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)