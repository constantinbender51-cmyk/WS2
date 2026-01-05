import os
import sys
import time
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
# 1. Environment & Auth
REPO_OWNER = "constantinbender51-cmyk"
REPO_NAME = "Models"
GITHUB_PAT = os.getenv("PAT") or os.getenv("GITHUB_TOKEN") # Ensure this is set!

# 2. Backtest Settings
INITIAL_CAPITAL = 10000.0
LEVERAGE = 2.0  # Matches your live bot setting
START_DATE = "2023-01-01"
END_DATE = "2024-01-01"
WARMUP_CANDLES = 1000  # Need enough history to train the first probability map

# 3. Asset Mapping (Binance Spot Data -> Kraken Futures Logic)
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

# --- Strategy Class (Identical to Production) ---
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

# --- Backtester Engine ---
class RealBacktester:
    def __init__(self):
        self.strategies = {} # id -> Strategy
        self.data_store = {} # asset -> DataFrame
        self.portfolio = {
            "cash": INITIAL_CAPITAL,
            "holdings": {k: 0.0 for k in SYMBOL_MAP.keys()},
            "equity_history": [],
            "ts_history": []
        }
        self.instrument_map = {v: k for k, v in SYMBOL_MAP.items()} # kf_symbol -> binance_symbol

    def load_real_strategies(self):
        """Downloads configurations directly from your private GitHub repo."""
        if not GITHUB_PAT:
            print("CRITICAL: No GITHUB_PAT found. Cannot download strategies.")
            sys.exit(1)

        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/"
        headers = {
            "Authorization": f"Bearer {GITHUB_PAT}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        print(f"Connecting to GitHub ({REPO_OWNER}/{REPO_NAME})...")
        try:
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            files = resp.json()
            
            count = 0
            for f in files:
                if f['name'].endswith(".json"):
                    # Download raw content
                    raw_resp = requests.get(f['download_url'], headers=headers)
                    data = raw_resp.json()
                    
                    asset = data['asset']
                    tf = data['timeframe']
                    
                    # Ensure we have a map for this asset
                    if asset not in SYMBOL_MAP:
                        continue 

                    # Select best strategy (Top 1)
                    best_strat = data['strategy_union'][0]
                    
                    s = Strategy(asset, tf, best_strat)
                    self.strategies[s.id] = s
                    count += 1
            
            print(f"Successfully loaded {count} strategies from GitHub.")
            
        except Exception as e:
            print(f"Error loading strategies: {e}")
            sys.exit(1)

    def fetch_market_data(self):
        """Fetches real 15m data from Binance for all active assets."""
        active_assets = set(s.asset for s in self.strategies.values())
        print(f"Fetching market data for {len(active_assets)} assets...")
        
        start_ts = int(datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(datetime.strptime(END_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        
        for asset in active_assets:
            filename = f"data_{asset}_{START_DATE}_{END_DATE}.csv"
            
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                df['ts'] = pd.to_datetime(df['ts'])
            else:
                # Download
                print(f"Downloading {asset}...")
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
        rule = rule_map.get(timeframe)
        if not rule: return df_slice['price'].tolist()
        
        temp = df_slice.set_index('ts')
        # Close price resampling
        return temp['price'].resample(rule).last().dropna().tolist()

    def run(self):
        # Align timestamps (use intersection of all assets)
        common_idx = None
        for asset, df in self.data_store.items():
            if common_idx is None: common_idx = df.index
            else: common_idx = common_idx.intersection(df.index)
        
        # Limit to available data length (roughly)
        min_len = min(len(df) for df in self.data_store.values())
        print(f"Running simulation on {min_len} candles...")

        for i in range(WARMUP_CANDLES, min_len):
            # 1. Current timestamp & Price Vector
            current_prices = {a: self.data_store[a].iloc[i]['price'] for a in self.data_store}
            current_ts = self.data_store[list(self.data_store.keys())[0]].iloc[i]['ts']
            
            # 2. Update Equity (Mark to Market)
            equity = self.portfolio["cash"]
            for a, pos in self.portfolio["holdings"].items():
                if a in current_prices:
                    equity += pos * current_prices[a]
            
            self.portfolio["equity_history"].append(equity)
            self.portfolio["ts_history"].append(current_ts)

            if equity <= 0:
                print("Bankruptcy.")
                break

            # 3. Strategy Logic
            total_strats = len(self.strategies)
            if total_strats == 0: continue
            
            unit_size_usd = (equity * LEVERAGE) / total_strats
            net_targets = defaultdict(float) # asset -> target_usd_value

            for s in self.strategies.values():
                # Slice history (Simulate "now")
                # Optimization: grab last 1000 candles relative to 'i'
                start = max(0, i - 2000)
                subset = self.data_store[s.asset].iloc[start : i + 1]
                
                prices = self.resample(subset, s.timeframe)
                
                # Train & Predict
                s.train(prices)
                sig = s.predict(prices)
                
                # Virtual Position
                s.virtual_position = sig * unit_size_usd
                net_targets[s.asset] += s.virtual_position

            # 4. Execution Logic
            for asset, target_usd in net_targets.items():
                price = current_prices.get(asset)
                if not price: continue
                
                # Calc target contracts
                target_contracts = target_usd / price
                current_contracts = self.portfolio["holdings"][asset]
                
                delta = target_contracts - current_contracts
                
                # Simple friction/fee model
                if abs(delta * price) > 5.0: # Min trade $5
                    cost = abs(delta * price)
                    fee = cost * 0.0005 # 0.05% Fee
                    
                    self.portfolio["cash"] -= fee
                    # Buying (pos delta) costs cash, Selling (neg delta) adds cash
                    self.portfolio["cash"] -= (delta * price)
                    self.portfolio["holdings"][asset] += delta
            
            # Progress log
            if i % 500 == 0:
                print(f"[{current_ts}] Eq: ${equity:.2f}")

    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.portfolio["ts_history"], self.portfolio["equity_history"])
        plt.title(f"Octopus Real Backtest (Eq: ${self.portfolio['equity_history'][-1]:.2f})")
        plt.grid(True)
        plt.savefig("backtest_real.png")
        print("Result saved to backtest_real.png")

if __name__ == "__main__":
    bt = RealBacktester()
    bt.load_real_strategies()
    bt.fetch_market_data()
    bt.run()
    bt.plot()
