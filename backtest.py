import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
SYMBOL = 'BTCUSDT'
INTERVAL = '1d'
START_DATE = '2018-01-01'
CACHE_FILE = 'btc_data.csv'

# Optimization Metric Settings
RISK_FREE_RATE = 0.0      # Assuming 0% for simple crypto backtesting
ANNUALIZATION_FACTOR = 365 # Crypto trades 365 days a year

# Grid Search Ranges
SMA1_RANGE = range(10, 501, 10) 
SMA2_RANGE = range(10, 181, 10)
SMA3_RANGE = range(10, 181, 10) # Added 3rd SMA
STOP_LOSS_RANGE = np.arange(0.01, 0.105, 0.005) # 1% to 10% step 0.5%
LEVERAGE_RANGE = np.arange(1.0, 5.5, 0.5)       # 1x to 5x step 0.5

# ==========================================
# 1. DATA FETCHING
# ==========================================
def fetch_binance_data(symbol, interval, start_str):
    """
    Fetches historical klines from Binance API with pagination.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    
    dt_obj = datetime.strptime(start_str, '%Y-%m-%d')
    start_ts = int(dt_obj.timestamp() * 1000)
    
    all_data = []
    current_start = start_ts
    
    print(f"Fetching {symbol} data from {start_str}...")
    
    while True:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'limit': 1000
        }
        
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if not isinstance(data, list) or len(data) == 0:
                break
                
            all_data.extend(data)
            current_start = data[-1][6] + 1
            time.sleep(0.05) # Slight delay to be kind to API
            
            last_date = datetime.fromtimestamp(data[-1][0]/1000).strftime('%Y-%m-%d')
            print(f"Fetched up to {last_date}...", end='\r')
            
            if data[-1][6] > time.time() * 1000:
                break
                
        except Exception as e:
            print(f"Connection error: {e}")
            break
            
    print(f"\nTotal records fetched: {len(all_data)}")
    
    df = pd.DataFrame(all_data, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
        'Close Time', 'Quote Asset Volume', 'Number of Trades', 
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ])
    
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    df['Date'] = pd.to_datetime(df['Open Time'], unit='ms')
    
    return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

def get_data():
    if os.path.exists(CACHE_FILE):
        print(f"Loading data from {CACHE_FILE}...")
        df = pd.read_csv(CACHE_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        df = fetch_binance_data(SYMBOL, INTERVAL, START_DATE)
        df.to_csv(CACHE_FILE, index=False)
    return df

# ==========================================
# 2. VECTORIZED SHARPE ENGINE
# ==========================================
def run_grid_search(df):
    print("Pre-calculating indicators and returns matrices...")
    
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    
    # -----------------------------------
    # A. Pre-calculate SMAs
    # -----------------------------------
    # Collect all unique periods needed for SMA1, SMA2, and SMA3
    all_periods = sorted(list(set(list(SMA1_RANGE) + list(SMA2_RANGE) + list(SMA3_RANGE))))
    smas = {}
    
    for p in all_periods:
        smas[p] = df['Open'].rolling(window=p).mean().values
        
    # -----------------------------------
    # B. Pre-calculate Base Returns (Unleveraged)
    # -----------------------------------
    long_returns_map = {}
    short_returns_map = {}
    
    # Standard daily return if no stop is hit (Arithmetic Returns)
    # Arithmetic is standard for Sharpe; Log returns are for compounding.
    std_long_ret = (closes - opens) / opens
    std_short_ret = (opens - closes) / opens
    
    for s in STOP_LOSS_RANGE:
        # Long Logic
        stop_price_long = opens * (1 - s)
        hit_long_sl = lows <= stop_price_long
        long_returns_map[s] = np.where(hit_long_sl, -s, std_long_ret)
        
        # Short Logic
        stop_price_short = opens * (1 + s)
        hit_short_sl = highs >= stop_price_short
        short_returns_map[s] = np.where(hit_short_sl, -s, std_short_ret)

    # -----------------------------------
    # C. Grid Search Loop
    # -----------------------------------
    best_sharpe = -999.0
    best_params = {}
    
    total_iter = len(SMA1_RANGE) * len(SMA2_RANGE) * len(SMA3_RANGE)
    print(f"Starting Sharpe Optimization over {total_iter} SMA triplets...")
    
    count = 0
    start_time = time.time()
    
    for sma1_p in SMA1_RANGE:
        sma1_arr = smas[sma1_p]
        
        for sma2_p in SMA2_RANGE:
            sma2_arr = smas[sma2_p]
            
            for sma3_p in SMA3_RANGE:
                # Redundancy check: If periods are identical, logic is same as 2 SMAs, but we allow it for completeness.
                # Optimization: Condition is "Above All". Order doesn't matter, but ranges differ.
                
                sma3_arr = smas[sma3_p]
                count += 1
                
                # 1. Calculate Signals
                # Long: Open > SMA1 & Open > SMA2 & Open > SMA3
                mask_long = (
                    (opens > sma1_arr) & 
                    (opens > sma2_arr) & 
                    (opens > sma3_arr)
                ).astype(int)
                
                # Short: Open < SMA1 & Open < SMA2 & Open < SMA3
                mask_short = (
                    (opens < sma1_arr) & 
                    (opens < sma2_arr) & 
                    (opens < sma3_arr)
                ).astype(int)
                
                # Skip if minimal activity
                if np.sum(mask_long) + np.sum(mask_short) < 10:
                    continue

                # 2. Loop Stop Losses
                for s_val in STOP_LOSS_RANGE:
                    
                    # Base Strategy Returns (Leverage = 1)
                    r_l = long_returns_map[s_val]
                    r_s = short_returns_map[s_val]
                    
                    # Daily return vector (0.0 if Flat)
                    base_daily_rets = (mask_long * r_l) + (mask_short * r_s)
                    
                    mean_ret = np.mean(base_daily_rets)
                    std_ret = np.std(base_daily_rets)
                    
                    if std_ret == 0:
                        current_sharpe = -999.0
                    else:
                        # Annualized Sharpe Formula
                        current_sharpe = np.sqrt(ANNUALIZATION_FACTOR) * (mean_ret / std_ret)
                    
                    # 3. Loop Leverage (Check for Busts)
                    min_daily_ret = np.min(base_daily_rets)
                    
                    for lev in LEVERAGE_RANGE:
                        # Check Bankruptcy
                        if min_daily_ret * lev <= -1.0:
                            continue
                        
                        if current_sharpe > best_sharpe:
                            best_sharpe = current_sharpe
                            best_params = {
                                'SMA1': sma1_p,
                                'SMA2': sma2_p,
                                'SMA3': sma3_p,
                                'StopLoss': round(s_val * 100, 2),
                                'Leverage': lev,
                                'Sharpe': round(current_sharpe, 4),
                                'AvgDailyRet': round(mean_ret * lev * 100, 4)
                            }

                if count % 500 == 0:
                    print(f"Processed {count}/{total_iter}... Best Sharpe: {best_params.get('Sharpe', -999)}", end='\r')

    end_time = time.time()
    print(f"\n\nOptimization Complete in {end_time - start_time:.2f} seconds.")
    return best_params

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    df = get_data()
    print(f"Data range: {df['Date'].min()} to {df['Date'].max()}")
    print("------------------------------------------------")
    
    result = run_grid_search(df)
    
    print("------------------------------------------------")
    print("OPTIMIZATION RESULTS (Max Sharpe)")
    print("------------------------------------------------")
    print(f"Best Sharpe Ratio: {result['Sharpe']}")
    print(f"Parameters:")
    print(f"  SMA 1 Period:    {result['SMA1']}")
    print(f"  SMA 2 Period:    {result['SMA2']}")
    print(f"  SMA 3 Period:    {result['SMA3']}")
    print(f"  Stop Loss:       {result['StopLoss']}%")
    print(f"  Leverage:        {result['Leverage']}x")
    print("------------------------------------------------")
    print("Note: If Risk-Free Rate is 0%, leverage does not increase Sharpe")
    print("unless it causes bankruptcy. The Lowest valid leverage was selected.")
