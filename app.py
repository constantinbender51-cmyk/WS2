import os
import sys
import time
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta

# --- Configuration ---
# 1. Environment & Auth
REPO_OWNER = "constantinbender51-cmyk"
REPO_NAME = "Models"
GITHUB_PAT = os.getenv("PAT") or os.getenv("GITHUB_TOKEN")

# 2. Data Scope (Full History)
# We fetch ALL data from 2020 to Now.
# The script will calculate the 70% cutoff automatically.
START_DATE = "2020-01-01" 
END_DATE = datetime.now().strftime("%Y-%m-%d") # Today

# 3. Backtest Settings
INITIAL_CAPITAL = 10000.0
LEVERAGE = 2.0
TRAIN_SPLIT = 0.70  # First 70% is Training, last 30% is Unseen (Trading)

# 4. Asset Mapping
SYMBOL_MAP = {
    "BTCUSDT": "ff_xbtusd_260327",
    "ETHUSDT": "pf_ethusd",
    "SOLUSDT": "pf_solusd",
    "BNBUSDT": "pf_bnbusd",
    "XRPUSDT": "pf_xrpusd",
    "ADAUSDT": "pf_adausd",
    "DOGEUSDT": "pf_dogeusd",
    "AVAXUSDT": "pf_avaxusd",
    "DOTUSDT": "pf_dotusd",
    "LINKUSDT": "pf_linkusd",
}

# --- Strategy Logic (Standard) ---
class Strategy:
    def __init__(self, asset: str, timeframe: str, config: dict):
        self.asset = asset
        self.timeframe = timeframe
        self.config = config
        self.bucket_size = config['bucket_size']
        self.seq_len = config['seq_len']
        self.model_type = config['model_type']
        
        self.virtual_position = 0.0
        self.abs_map = defaultdict(Counter)
        self.der_map = defaultdict(Counter)
        self.id = f"{asset}_{timeframe}"

    def train(self, prices):
        """Builds the Probability Map from historical prices."""
        buckets = [self._get_bucket(p) for p in prices]
        if len(buckets) < self.seq_len + 10: return
        
        self.abs_map.clear()
        self.der_map.clear()
        
        for i in range(len(buckets) - self.seq_len):
            a_seq = tuple(buckets[i : i + self.seq_len])
            a_succ = buckets[i + self.seq_len]
            self.abs_map[a_seq][a_succ] += 1
            
            if i > 0:
                d_seq = tuple(buckets[j] - buckets[j-1] for j in range(i, i + self.seq_len))
                d_succ = buckets[i + self.seq_len] - buckets[i + self.seq_len - 1]
                self.der_map[d_seq][d_succ] += 1

    def predict(self, recent_prices) -> int:
        if len(recent_prices) < self.seq_len + 1: return 0
        buckets = [self._get_bucket(p) for p in recent_prices]
        window = buckets[-(self.seq_len + 1):] 
        
        a_seq = tuple(window[1:]) 
        d_seq = tuple(window[j] - window[j-1] for j in range(1, len(window)))
        last_val = window[-1]
        
        pred_bucket = last_val
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
        if price >= 0: return (int(price) // self.bucket_size) + 1
        else: return (int(price + 1) // self.bucket_size) - 1

# --- Out-of-Sample Backtester ---
class OOSBacktester:
    def __init__(self):
        self.strategies = {}
        self.data_store = {} 
        self.portfolio = {
            "cash": INITIAL_CAPITAL,
            "holdings": {k: 0.0 for k in SYMBOL_MAP.keys()},
            "equity_history": [],
            "ts_history": []
        }
        self.split_timestamp = None

    def load_strategies(self):
        """Downloads configs from GitHub."""
        if not GITHUB_PAT:
            print("CRITICAL: No GITHUB_PAT found.")
            sys.exit(1)

        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/"
        headers = {"Authorization": f"Bearer {GITHUB_PAT}"}
        
        try:
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            files = resp.json()
            
            count = 0
            for f in files:
                if f['name'].endswith(".json"):
                    data = requests.get(f['download_url'], headers=headers).json()
                    asset = data['asset']
                    tf = data['timeframe']
                    if asset in SYMBOL_MAP:
                        best_strat = data['strategy_union'][0]
                        s = Strategy(asset, tf, best_strat)
                        self.strategies[s.id] = s
                        count += 1
            print(f"Loaded {count} strategies.")
        except Exception as e:
            print(f"Error loading strategies: {e}")
            sys.exit(1)

    def fetch_full_history(self):
        """Fetches full history (2020-Now) to establish the split."""
        active_assets = set(s.asset for s in self.strategies.values())
        print(f"Fetching full history (2020 - Now) for {len(active_assets)} assets...")
        
        start_ts = int(datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(datetime.now().replace(tzinfo=timezone.utc).timestamp() * 1000)
        
        for asset in active_assets:
            filename = f"full_data_{asset}.csv"
            
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                df['ts'] = pd.to_datetime(df['ts'])
            else:
                print(f"Downloading {asset} (This may take a minute)...")
                all_candles = []
                current = start_ts
                while current < end_ts:
                    url = "https://api.binance.com/api/v3/klines"
                    params = {"symbol": asset, "interval": "15m", "limit": 1000, "startTime": current}
                    r = requests.get(url, params=params)
                    data = r.json()
                    if not data: break
                    all_candles.extend(data)
                    current = int(data[-1][6]) + 1
                    time.sleep(0.05)
                
                df = pd.DataFrame(all_candles, columns=['ot','o','h','l','c','v','ct','q','n','tb','tq','i'])
                df['ts'] = pd.to_datetime(df['ct'], unit='ms')
                df['price'] = df['c'].astype(float)
                df = df[['ts', 'price']]
                df.to_csv(filename, index=False)
            
            self.data_store[asset] = df

    def resample(self, df_slice, timeframe):
        if timeframe == "15m": return df_slice['price'].tolist()
        rule_map = {"30m": "30min", "60m": "1h", "240m": "4h", "1d": "1D"}
        temp = df_slice.set_index('ts')
        return temp['price'].resample(rule_map[timeframe]).last().dropna().tolist()

    def run_oos_test(self):
        # 1. Align Data
        common_idx = None
        for df in self.data_store.values():
            if common_idx is None: common_idx = df.index
            else: common_idx = common_idx.intersection(df.index)
        
        total_len = len(df) # Approximate length of aligned data
        split_idx = int(total_len * TRAIN_SPLIT)
        
        # Get the timestamp where the "Unseen" period begins
        # We assume all DataFrames are roughly aligned by row count now
        # Ideally, we grab the timestamp from the first asset at the split index
        first_asset = list(self.data_store.keys())[0]
        self.split_timestamp = self.data_store[first_asset].iloc[split_idx]['ts']
        
        print(f"--- SPLIT CALCULATION ---")
        print(f"Total Candles: {total_len}")
        print(f"Training Data: First {TRAIN_SPLIT*100}% (approx {split_idx} candles)")
        print(f"Unseen Data starts at: {self.split_timestamp}")
        print(f"Backtesting ONLY on Unseen Data...")
        print(f"-------------------------")

        # 2. Loop strictly through the Unseen portion
        # We start 'i' at split_idx
        for i in range(split_idx, total_len):
            
            current_ts = self.data_store[first_asset].iloc[i]['ts']
            
            # --- A. Mark to Market ---
            equity = self.portfolio["cash"]
            current_prices = {}
            for asset, df in self.data_store.items():
                # Safety check for index bounds
                if i < len(df):
                    price = df.iloc[i]['price']
                    current_prices[asset] = price
                    equity += self.portfolio["holdings"][asset] * price
            
            self.portfolio["equity_history"].append(equity)
            self.portfolio["ts_history"].append(current_ts)
            
            if equity <= 0:
                print(f"Bankruptcy at {current_ts}")
                break

            # --- B. Strategy Updates ---
            total_strats = len(self.strategies)
            unit_size = (equity * LEVERAGE) / total_strats if total_strats > 0 else 0
            net_targets = defaultdict(float)

            for s in self.strategies.values():
                # Walk-Forward Logic:
                # The model is allowed to "see" history up to 'i'.
                # Even though we are in the unseen period, the model continually updates 
                # its map with the data it just experienced (just like in live trading).
                
                # Optimization: Limit lookback for training (last 3000 candles is plenty for map building)
                start_lookback = max(0, i - 3000)
                subset = self.data_store[s.asset].iloc[start_lookback : i + 1]
                
                prices = self.resample(subset, s.timeframe)
                s.train(prices)
                sig = s.predict(prices)
                
                s.virtual_position = sig * unit_size
                net_targets[s.asset] += s.virtual_position

            # --- C. Execution ---
            for asset, target_usd in net_targets.items():
                if asset not in current_prices: continue
                price = current_prices[asset]
                
                current_qty = self.portfolio["holdings"][asset]
                target_qty = target_usd / price
                delta = target_qty - current_qty
                
                trade_value = abs(delta * price)
                
                # Minimum trade size filter (simulates dust limits)
                if trade_value > 10.0:
                    fee = trade_value * 0.0005 # 0.05%
                    self.portfolio["cash"] -= fee
                    self.portfolio["cash"] -= (delta * price)
                    self.portfolio["holdings"][asset] += delta

            # Log periodically
            if i % 1000 == 0:
                print(f"[{current_ts}] Eq: ${equity:.2f}")

    def report(self):
        equity = self.portfolio["equity_history"]
        if not equity:
            print("No trades executed.")
            return

        final_eq = equity[-1]
        ret = (final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        
        print("\n--- Out-of-Sample Results (30% Unseen) ---")
        print(f"Start Date:   {self.split_timestamp}")
        print(f"End Date:     {self.portfolio['ts_history'][-1]}")
        print(f"Initial Cap:  ${INITIAL_CAPITAL:.2f}")
        print(f"Final Cap:    ${final_eq:.2f}")
        print(f"Return:       {ret:.2f}%")

        plt.figure(figsize=(12, 6))
        plt.plot(self.portfolio["ts_history"], equity, label="Equity (OOS)")
        plt.axvline(x=self.split_timestamp, color='r', linestyle='--', label="Unseen Data Start")
        plt.title(f"Performance on Unseen Data (Last 30%)")
        plt.legend()
        plt.grid(True)
        plt.savefig("backtest_oos.png")
        print("Chart saved to backtest_oos.png")

if __name__ == "__main__":
    bt = OOSBacktester()
    bt.load_strategies()
    bt.fetch_full_history()
    bt.run_oos_test()
    bt.report()
