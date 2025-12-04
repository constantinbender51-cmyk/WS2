import io
import base64
from flask import Flask, render_template_string
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import itertools

app = Flask(__name__)

# -----------------------------------------------------------------------------
# 1. DATA FETCHING
# -----------------------------------------------------------------------------
def fetch_binance_data(symbol="BTCUSDT", interval="1d", start_date="2018-01-01"):
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    
    all_data = []
    limit = 1000
    current_start = start_ts
    
    print(f"Fetching {symbol} data from {start_date}...")
    
    while True:
        params = {"symbol": symbol, "interval": interval, "startTime": current_start, "limit": limit}
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            if not isinstance(data, list) or len(data) == 0: break
            all_data.extend(data)
            current_start = data[-1][6] + 1
            if len(data) < limit: break
            time.sleep(0.05)
        except Exception as e:
            print(f"Error: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"
    ])
    
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("date", inplace=True)
    return df[["open", "close"]]

# -----------------------------------------------------------------------------
# 2. GRID SEARCH OPTIMIZATION
# -----------------------------------------------------------------------------
def run_heavy_grid_search(df):
    """
    Performs a grid search over 4 dimensions: SMA_Z, SMA_O, X%, Y%.
    Uses Numpy for speed.
    """
    print("\n--- Pre-calculating Data & SMAs ---")
    
    # 1. Define Parameter Spaces
    sma_periods = [1, 2, 4, 8, 16, 32, 64, 128]
    # Scanning X and Y from 0% to 5% in 0.5% steps
    thresholds = np.arange(0.00, 0.051, 0.005) 
    
    # 2. Pre-calculate all SMAs and convert to Numpy
    # We use 'Close' for SMA calculation, but shift it by 1 so we have "Yesterday's SMA"
    # This prevents lookahead bias when comparing against Today's Open.
    sma_dict = {}
    for period in sma_periods:
        # Calculate SMA on Close, then shift 1 to align with next day's Open
        sma_series = df['close'].rolling(window=period).mean().shift(1)
        sma_dict[period] = sma_series.to_numpy()

    # Base arrays
    opens = df['open'].to_numpy()
    
    # Calculate Market Returns (Open-to-Open or Close-to-Close log returns)
    # Using Close-to-Close for standard portfolio simulation
    market_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0).to_numpy()
    
    # Track Best Results
    best_sharpe = -np.inf
    best_params = None
    best_equity_curve = None
    
    # Generate all combinations
    # We use combinations_with_replacement for SMAs if order didn't matter, 
    # but here order matters because X applies to Z and Y applies to O.
    # To save time, we can assume symmetry checks, but let's do full product for correctness.
    sma_combinations = list(itertools.product(sma_periods, sma_periods))
    thresh_combinations = list(itertools.product(thresholds, thresholds))
    
    total_iterations = len(sma_combinations) * len(thresh_combinations)
    print(f"Starting Grid Search: {total_iterations} combinations...")
    
    start_time = time.time()
    counter = 0
    
    # 3. The Loop (Optimized with Numpy)
    # We iterate SMAs first, as pulling the array is the "expensive" part
    for z, o in sma_combinations:
        sma_z_arr = sma_dict[z]
        sma_o_arr = sma_dict[o]
        
        # Valid mask: where both SMAs are not NaN
        valid_mask = (~np.isnan(sma_z_arr)) & (~np.isnan(sma_o_arr))
        
        for x, y in thresh_combinations:
            counter += 1
            
            # --- Vectorized Logic ---
            
            # Long: Open > SMA_Z*(1+x) AND Open > SMA_O*(1+y)
            # We must handle NaNs (comparisons with NaN return False usually, but good to be safe)
            long_condition = (opens > sma_z_arr * (1 + x)) & (opens > sma_o_arr * (1 + y))
            
            # Short: Open < SMA_Z*(1-x) AND Open < SMA_O*(1-y)
            short_condition = (opens < sma_z_arr * (1 - x)) & (opens < sma_o_arr * (1 - y))
            
            # Position: 1 for Long, -1 for Short, 0 otherwise
            # Initialize with zeros
            positions = np.zeros_like(opens)
            positions[long_condition] = 1
            positions[short_condition] = -1
            
            # Strategy Returns = Position * Market Returns
            strat_returns = positions * market_returns
            
            # Metrics
            # Filter out the initial NaNs for accurate mean/std
            active_returns = strat_returns[valid_mask]
            
            if len(active_returns) == 0: continue
                
            mean_ret = np.mean(active_returns)
            std_ret = np.std(active_returns)
            
            if std_ret < 1e-9:
                sharpe = 0
            else:
                sharpe = (mean_ret / std_ret) * np.sqrt(365)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (z, o, x, y)
                best_equity_curve = np.cumsum(strat_returns) # Log returns sum
                
            if counter % 5000 == 0:
                print(f"Processed {counter}/{total_iterations}...")

    end_time = time.time()
    print(f"\nSearch complete in {end_time - start_time:.2f} seconds.")
    
    return best_params, best_sharpe, best_equity_curve, market_returns

# -----------------------------------------------------------------------------
# WEB ROUTES
# -----------------------------------------------------------------------------
@app.route('/')
def index():
    # Run the backtest
    SYMBOL = "BTCUSDT"
    df = fetch_binance_data(SYMBOL)
    best_params, best_sharpe, best_curve, market_returns = run_heavy_grid_search(df)
    
    z, o, x, y = best_params
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Convert log returns to cumulative percentage
    strat_cum = np.exp(best_curve)
    market_cum = np.exp(np.cumsum(market_returns))
    
    # Align lengths
    dates = df.index
    
    plt.plot(dates, market_cum, label="Buy & Hold (BTC)", color='gray', alpha=0.5)
    plt.plot(dates, np.exp(np.cumsum(np.pad(best_curve, (0,0)))), label=f"Strategy (Sharpe: {best_sharpe:.2f})", color='blue')
    
    plt.title(f"Best Grid Search Result: SMA({z}) +/- {x:.1%} & SMA({o}) +/- {y:.1%}")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Encode the image to base64
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Create HTML with the plot
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Visualization</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; }
            .results { background: #f5f5f5; padding: 20px; border-radius: 5px; margin: 20px 0; }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Backtest Visualization</h1>
        <div class="results">
            <h2>Optimal Results</h2>
            <p><strong>SMA Period Z:</strong> {{ z }}</p>
            <p><strong>SMA Period O:</strong> {{ o }}</p>
            <p><strong>Threshold X:</strong> {{ x_percent }}</p>
            <p><strong>Threshold Y:</strong> {{ y_percent }}</p>
            <p><strong>Sharpe Ratio:</strong> {{ sharpe }}</p>
        </div>
        <div>
            <h2>Equity Curve</h2>
            <img src="data:image/png;base64,{{ plot_data }}" alt="Backtest Plot">
        </div>
        <p><em>Note: This backtest runs each time the page is loaded and may take 10-20 seconds.</em></p>
    </body>
    </html>
    """
    
    return render_template_string(html_template, 
                                  z=z, o=o, 
                                  x_percent=f"{x:.2%}", 
                                  y_percent=f"{y:.2%}", 
                                  sharpe=f"{best_sharpe:.4f}", 
                                  plot_data=plot_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)