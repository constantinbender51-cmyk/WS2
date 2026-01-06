import os
import sys
import time
import json
import requests
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta

# --- Configuration ---
# 1. Environment & Auth
REPO_OWNER = "constantinbender51-cmyk"
REPO_NAME = "Models"
GITHUB_PAT = os.getenv("PAT") or os.getenv("GITHUB_TOKEN")

# 2. Data Scope (Full History)
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

def delayed_print(text):
    """Custom print with 0.1s delay as requested."""
    print(text)
    time.sleep(0.1)

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
            "ts_history": [],
            "returns_history": [],
            "drawdown_history": []
        }
        self.metrics = {
            "wins": 0,
            "losses": 0,
            "total_intervals": 0,
            "correct_predictions": 0,
            "total_predictions": 0,
            "gross_profit": 0.0,
            "gross_loss": 0.0
        }
        self.split_timestamp = None

    def load_strategies(self):
        """Downloads configs from GitHub."""
        if not GITHUB_PAT:
            delayed_print("CRITICAL: No GITHUB_PAT found.")
            sys.exit(1)

        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/"
        headers = {"Authorization": f"Bearer {GITHUB_PAT}"}
        
        try:
            delayed_print("Connecting to GitHub...")
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
            delayed_print(f"Loaded {count} strategies.")
        except Exception as e:
            delayed_print(f"Error loading strategies: {e}")
            sys.exit(1)

    def fetch_full_history(self):
        """Fetches full history (2020-Now) to establish the split."""
        active_assets = set(s.asset for s in self.strategies.values())
        delayed_print(f"Fetching full history (2020 - Now) for {len(active_assets)} assets...")
        
        start_ts = int(datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(datetime.now().replace(tzinfo=timezone.utc).timestamp() * 1000)
        
        for asset in active_assets:
            filename = f"full_data_{asset}.csv"
            
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                df['ts'] = pd.to_datetime(df['ts'])
            else:
                delayed_print(f"Downloading {asset} (This may take a minute)...")
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
        
        first_asset = list(self.data_store.keys())[0]
        self.split_timestamp = self.data_store[first_asset].iloc[split_idx]['ts']
        
        delayed_print(f"--- SPLIT CALCULATION ---")
        delayed_print(f"Total Candles: {total_len}")
        delayed_print(f"Training Data: First {TRAIN_SPLIT*100}% (approx {split_idx} candles)")
        delayed_print(f"Unseen Data starts at: {self.split_timestamp}")
        delayed_print(f"Backtesting ONLY on Unseen Data...")
        delayed_print(f"-------------------------")

        prev_equity = self.portfolio["cash"]
        peak_equity = prev_equity
        prev_net_exposure = 0.0

        # 2. Loop strictly through the Unseen portion
        for i in range(split_idx, total_len):
            
            current_ts = self.data_store[first_asset].iloc[i]['ts']
            
            # --- A. Mark to Market ---
            equity = self.portfolio["cash"]
            current_prices = {}
            for asset, df in self.data_store.items():
                if i < len(df):
                    price = df.iloc[i]['price']
                    current_prices[asset] = price
                    equity += self.portfolio["holdings"][asset] * price
            
            # Metrics: Returns & Drawdown
            step_return = equity - prev_equity
            self.portfolio["equity_history"].append(equity)
            self.portfolio["ts_history"].append(current_ts)
            self.portfolio["returns_history"].append(step_return)
            
            if equity > peak_equity: peak_equity = equity
            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            self.portfolio["drawdown_history"].append(drawdown)

            # Metrics: Win/Loss Counts (Interval based)
            self.metrics["total_intervals"] += 1
            if step_return > 0:
                self.metrics["wins"] += 1
                self.metrics["gross_profit"] += step_return
            elif step_return < 0:
                self.metrics["losses"] += 1
                self.metrics["gross_loss"] += abs(step_return)

            # Metrics: Directional Accuracy
            # Compare previous exposure to current PnL direction
            if prev_net_exposure > 0 and step_return > 0: self.metrics["correct_predictions"] += 1
            elif prev_net_exposure < 0 and step_return < 0: self.metrics["correct_predictions"] += 1
            elif prev_net_exposure != 0: 
                # If we had exposure but PnL didn't match direction (loss)
                pass 
            
            if prev_net_exposure != 0:
                 self.metrics["total_predictions"] += 1
            
            if equity <= 0:
                delayed_print(f"Bankruptcy at {current_ts}")
                break

            prev_equity = equity

            # --- B. Strategy Updates ---
            total_strats = len(self.strategies)
            unit_size = (equity * LEVERAGE) / total_strats if total_strats > 0 else 0
            net_targets = defaultdict(float)
            current_total_exposure = 0.0

            for s in self.strategies.values():
                start_lookback = max(0, i - 3000)
                subset = self.data_store[s.asset].iloc[start_lookback : i + 1]
                
                prices = self.resample(subset, s.timeframe)
                s.train(prices)
                sig = s.predict(prices)
                
                s.virtual_position = sig * unit_size
                net_targets[s.asset] += s.virtual_position
                
                # Simple sum of all target USD exposures
                current_total_exposure += s.virtual_position

            prev_net_exposure = current_total_exposure

            # --- C. Execution ---
            for asset, target_usd in net_targets.items():
                if asset not in current_prices: continue
                price = current_prices[asset]
                
                current_qty = self.portfolio["holdings"][asset]
                target_qty = target_usd / price
                delta = target_qty - current_qty
                
                trade_value = abs(delta * price)
                
                if trade_value > 10.0:
                    fee = trade_value * 0.0005 # 0.05%
                    self.portfolio["cash"] -= fee
                    self.portfolio["cash"] -= (delta * price)
                    self.portfolio["holdings"][asset] += delta

            # Log periodically
            if i % 1000 == 0:
                delayed_print(f"[{current_ts}] Eq: ${equity:,.2f} | DD: {drawdown*100:.2f}%")

    def report(self):
        equity = self.portfolio["equity_history"]
        if not equity:
            delayed_print("No trades executed.")
            return

        final_eq = equity[-1]
        total_return = (final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL
        
        # Safe calculations
        returns_np = np.array(self.portfolio["returns_history"])
        # Filter out first zero if exists to avoid skew
        if len(returns_np) > 1: returns_np = returns_np[1:] 
        
        # Annualized metrics (Assuming 15m intervals -> 35040 intervals/year)
        intervals_per_year = 35040
        mean_return = np.mean(returns_np)
        std_return = np.std(returns_np)
        
        sharpe = 0.0
        if std_return > 0:
            sharpe = (mean_return / std_return) * np.sqrt(intervals_per_year)
            
        # Sortino
        downside_returns = returns_np[returns_np < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1.0
        sortino = 0.0
        if downside_std > 0:
            sortino = (mean_return / downside_std) * np.sqrt(intervals_per_year)

        # Max Drawdown
        max_dd = max(self.portfolio["drawdown_history"])
        
        # Profit Factor
        gross_p = self.metrics["gross_profit"]
        gross_l = self.metrics["gross_loss"]
        profit_factor = gross_p / gross_l if gross_l > 0 else float('inf')

        # Win Rate (Intervals)
        win_rate = (self.metrics["wins"] / self.metrics["total_intervals"]) * 100 if self.metrics["total_intervals"] > 0 else 0
        
        # Directional Accuracy (Active predictions)
        accuracy = (self.metrics["correct_predictions"] / self.metrics["total_predictions"]) * 100 if self.metrics["total_predictions"] > 0 else 0

        # Text Only Report
        print("\n" + "="*40)
        print("   OUT-OF-SAMPLE PERFORMANCE REPORT")
        print("="*40)
        print(f"Start Date:          {self.split_timestamp}")
        print(f"End Date:            {self.portfolio['ts_history'][-1]}")
        print("-" * 40)
        print(f"Initial Capital:     ${INITIAL_CAPITAL:,.2f}")
        print(f"Final Equity:        ${final_eq:,.2f}")
        print(f"Total Return:        {total_return*100:.2f}%")
        print("-" * 40)
        print(f"Annualized Sharpe:   {sharpe:.4f}")
        print(f"Annualized Sortino:  {sortino:.4f}")
        print(f"Max Drawdown:        {max_dd*100:.2f}%")
        print(f"Profit Factor:       {profit_factor:.2f}")
        print("-" * 40)
        print(f"Win Rate (Interval): {win_rate:.2f}%")
        print(f"Direction Accuracy:  {accuracy:.2f}%")
        print(f"Total Intervals:     {self.metrics['total_intervals']}")
        print("="*40 + "\n")

if __name__ == "__main__":
    bt = OOSBacktester()
    bt.load_strategies()
    bt.fetch_full_history()
    bt.run_oos_test()
    bt.report()