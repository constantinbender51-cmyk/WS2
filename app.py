import sys
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from collections import Counter, defaultdict
from typing import List, Tuple

# --- Configuration ---
# 70% Train, 30% Test
TRAIN_SPLIT = 0.7 
ASSET = "ETHUSDT"
TIMEFRAME = "1h"  # Binance format for 60m
START_YEAR = 2020
REPO_OWNER = "constantinbender51-cmyk"
REPO_NAME = "Models"

# --- Helper: Rate Limited Print ---
def rprint(msg):
    """Prints message with a slight delay for rate-limited consoles."""
    print(msg)
    sys.stdout.flush()
    time.sleep(0.1)

# --- Class: Strategy (Adapted from Octopus) ---
class Strategy:
    def __init__(self, asset: str, timeframe: str, config: dict):
        self.asset = asset
        self.timeframe = timeframe
        self.bucket_size = config.get('bucket_size', 50) # Fallback default
        self.seq_len = config.get('seq_len', 6)
        self.model_type = config.get('model_type', 'Combined')
        
        self.abs_map = defaultdict(Counter)
        self.der_map = defaultdict(Counter)

    def train(self, prices: List[float]):
        """Builds probability maps from price history."""
        if not prices: return

        buckets = [self._get_bucket(p) for p in prices]
        
        if len(buckets) < self.seq_len + 10:
            return 
            
        self.abs_map.clear()
        self.der_map.clear()
        
        # Build maps (Pattern Recognition)
        for i in range(len(buckets) - self.seq_len):
            # Absolute Sequence
            a_seq = tuple(buckets[i : i + self.seq_len])
            a_succ = buckets[i + self.seq_len]
            self.abs_map[a_seq][a_succ] += 1
            
            # Derivative Sequence
            if i > 0:
                d_seq = tuple(buckets[j] - buckets[j-1] for j in range(i, i + self.seq_len))
                d_succ = buckets[i + self.seq_len] - buckets[i + self.seq_len - 1]
                self.der_map[d_seq][d_succ] += 1

    def predict(self, recent_prices: List[float]) -> int:
        """Returns 1 (Buy), -1 (Sell), 0 (Neutral)."""
        if len(recent_prices) < self.seq_len + 1:
            return 0
            
        buckets = [self._get_bucket(p) for p in recent_prices]
        window = buckets[-(self.seq_len + 1):] 
        
        # Extract patterns
        a_seq = tuple(window[1:]) 
        d_seq = tuple(window[j] - window[j-1] for j in range(1, len(window)))
        last_val = window[-1]
        
        pred_bucket = last_val
        
        # Model Logic
        if self.model_type == "Absolute":
            if a_seq in self.abs_map:
                pred_bucket = self.abs_map[a_seq].most_common(1)[0][0]
        elif self.model_type == "Derivative":
            if d_seq in self.der_map:
                change = self.der_map[d_seq].most_common(1)[0][0]
                pred_bucket = last_val + change
        elif self.model_type == "Combined":
            abs_cand = self.abs_map.get(a_seq, Counter())
            der_cand = self.der_map.get(d_seq, Counter())
            
            poss = set(abs_cand.keys())
            for c in der_cand.keys(): poss.add(last_val + c)
            
            best, max_s = last_val, -1
            for v in poss:
                s = abs_cand[v] + der_cand[v - last_val]
                if s > max_s: max_s, best = s, v
            pred_bucket = best

        if pred_bucket > last_val: return 1
        elif pred_bucket < last_val: return -1
        else: return 0

    def _get_bucket(self, price: float) -> int:
        if price >= 0:
            return (int(price) // self.bucket_size) + 1
        else:
            return (int(price + 1) // self.bucket_size) - 1

# --- Data Fetching ---
def fetch_historical_data(symbol: str, interval: str, start_year: int) -> List[Tuple[int, float]]:
    rprint(f"Fetching {interval} data for {symbol} starting from {start_year}...")
    
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime(start_year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    
    all_klines = []
    
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "limit": 1000
        }
        try:
            resp = requests.get(base_url, params=params)
            data = resp.json()
            
            if not isinstance(data, list) or not data:
                break
                
            # [0]=OpenTime, [4]=ClosePrice
            for k in data:
                all_klines.append((int(k[0]), float(k[4])))
            
            rprint(f"Loaded {len(all_klines)} candles... Last: {datetime.fromtimestamp(data[-1][0]/1000, tz=timezone.utc)}")
            
            if len(data) < 1000:
                break
                
            # Next batch starts 1ms after last close
            start_ts = int(data[-1][6]) + 1
            time.sleep(0.1) # Be nice to Binance API
            
        except Exception as e:
            rprint(f"Error fetching data: {e}")
            break
            
    return all_klines

# --- GitHub Config Fetcher ---
def fetch_github_config(owner: str, repo: str) -> dict:
    rprint("Attempting to fetch configuration from GitHub...")
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/"
    
    # Try to find a file matching ETH and 60m/1h
    target_keywords = ["ETH", "60m"]
    
    try:
        # Note: This requests the public repo contents. 
        # If private, this will fail without a token, and we fall back to defaults.
        resp = requests.get(api_url)
        if resp.status_code != 200:
            raise Exception("Repo not accessible (likely private or rate limited)")
            
        files = resp.json()
        target_file_url = None
        
        for f in files:
            name = f['name']
            if name.endswith(".json") and all(k in name for k in target_keywords):
                target_file_url = f['download_url']
                rprint(f"Found configuration file: {name}")
                break
        
        if target_file_url:
            cfg_resp = requests.get(target_file_url)
            full_config = cfg_resp.json()
            # Extract the best strategy from the union
            if 'strategy_union' in full_config and len(full_config['strategy_union']) > 0:
                return full_config['strategy_union'][0]
    except Exception as e:
        rprint(f"Could not fetch/parse GitHub config: {e}")
    
    rprint("USING FALLBACK CONFIGURATION (bucket=20, seq=6, Combined)")
    return {"bucket_size": 20, "seq_len": 6, "model_type": "Combined"}

# --- Main Backtest Logic ---
def run_backtest():
    # 1. Fetch Data
    raw_data = fetch_historical_data(ASSET, TIMEFRAME, START_YEAR)
    if not raw_data:
        rprint("No data found. Exiting.")
        return

    prices = [x[1] for x in raw_data]
    timestamps = [x[0] for x in raw_data]
    
    total_len = len(prices)
    split_idx = int(total_len * TRAIN_SPLIT)
    
    train_prices = prices[:split_idx]
    test_prices = prices[split_idx:]
    test_timestamps = timestamps[split_idx:]
    
    rprint(f"Total Data Points: {total_len}")
    rprint(f"Training Set (70%): {len(train_prices)} candles")
    rprint(f"Testing Set (30%): {len(test_prices)} candles (Unseen)")
    
    # 2. Setup Strategy
    config = fetch_github_config(REPO_OWNER, REPO_NAME)
    strat = Strategy(ASSET, TIMEFRAME, config)
    
    # 3. Train (Static - Once)
    rprint("Training model on In-Sample data (Static)...")
    start_train = time.time()
    strat.train(train_prices)
    rprint(f"Training complete in {time.time() - start_train:.2f}s")
    
    # 4. Walk-Forward Test (No re-training)
    rprint("Starting Walk-Forward Validation on Out-of-Sample data...")
    
    position = 0 # 1 (Long), -1 (Short), 0 (Flat)
    balance = 1000.0 # Starting hypothetical equity
    initial_balance = balance
    
    wins = 0
    losses = 0
    neutrals = 0
    
    # For Sharpe
    daily_returns = []
    current_day_start_equity = balance
    last_day_ts = test_timestamps[0]
    
    # We need a rolling window that starts with the end of training data
    # to predict the first point of test data
    history_window = train_prices[-(strat.seq_len + 5):]
    
    log_interval = len(test_prices) // 10 # Log 10 times during process
    
    for i, (price, ts) in enumerate(zip(test_prices, test_timestamps)):
        # 1. Predict next move based on history
        # (We simulate that 'price' has not happened yet for the prediction, 
        # but we need it to verify the result)
        
        signal = strat.predict(history_window)
        
        # 2. Execution Logic (Simplified: Close to Close)
        # If we had a position from previous step, calculate PnL change
        if i > 0:
            prev_price = test_prices[i-1]
            pnl_pct = (price - prev_price) / prev_price
            
            # Apply PnL based on position held coming INTO this candle
            if position != 0:
                # Fee assumption: 0.05% per trade if flipping
                # Here we simplify: Just raw PnL of holding
                trade_res = position * pnl_pct
                balance_change = balance * trade_res
                balance += balance_change
                
                # Accuracy tracking
                if trade_res > 0: wins += 1
                elif trade_res < 0: losses += 1
        
        # 3. Update Position for NEXT candle
        # In a static backtest, we trust the model's signal
        # 1 = Buy, -1 = Sell, 0 = Exit/Flat
        if signal != 0:
            position = signal
        else:
            position = 0
            neutrals += 1
            
        # 4. Update History Window (Slide)
        history_window.append(price)
        if len(history_window) > strat.seq_len + 10:
            history_window.pop(0)
            
        # 5. Sharpe Tracking (Daily)
        # Approx: 86400000 ms per day
        if ts - last_day_ts >= 86400000:
            day_ret = (balance - current_day_start_equity) / current_day_start_equity
            daily_returns.append(day_ret)
            current_day_start_equity = balance
            last_day_ts = ts
            
        # Logging progress
        if i % log_interval == 0:
            dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime('%Y-%m-%d')
            rprint(f"Progress {i}/{len(test_prices)} [{dt}] | Equity: ${balance:.2f}")

    # --- Final Metrics ---
    total_trades = wins + losses
    accuracy = (wins / total_trades * 100) if total_trades > 0 else 0
    total_pnl = balance - initial_balance
    total_return_pct = (total_pnl / initial_balance) * 100
    
    # Buy and Hold Benchmark
    bnh_return_pct = ((test_prices[-1] - test_prices[0]) / test_prices[0]) * 100
    
    # Sharpe
    if daily_returns:
        mean_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns)
        sharpe = (mean_ret / std_ret) * np.sqrt(365) if std_ret != 0 else 0
    else:
        sharpe = 0
        
    rprint("\n" + "="*40)
    rprint("       BACKTEST RESULTS (STATIC)       ")
    rprint("="*40)
    rprint(f"Asset:          {ASSET}")
    rprint(f"Config Source:  {REPO_NAME} (Bucket: {strat.bucket_size}, Seq: {strat.seq_len})")
    rprint(f"Data Points:    {len(test_prices)} hours (Unseen)")
    rprint("-" * 40)
    rprint(f"Final Balance:  ${balance:.2f} (Start: $1000)")
    rprint(f"Total PnL:      ${total_pnl:.2f}")
    rprint(f"Return:         {total_return_pct:.2f}%")
    rprint(f"Buy & Hold:     {bnh_return_pct:.2f}%")
    rprint("-" * 40)
    rprint(f"Accuracy:       {accuracy:.2f}% ({wins} W / {losses} L)")
    rprint(f"Sharpe Ratio:   {sharpe:.2f}")
    rprint("="*40)

if __name__ == "__main__":
    try:
        run_backtest()
    except KeyboardInterrupt:
        rprint("\nBacktest cancelled by user.")