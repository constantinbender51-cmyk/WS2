import requests
import pandas as pd
import numpy as np
import time
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

# --- Configuration ---
REPO_OWNER = "constantinbender51-cmyk"
REPO_NAME = "Models"
GITHUB_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/"
BACKTEST_DAYS = 90  # 3 Months

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Backtester")

# --- Strategy Classes (Ported from Octopus) ---

class SubStrategy:
    def __init__(self, strategy_data: dict):
        config = strategy_data.get('config', {})
        params = strategy_data.get('trained_parameters', {})
        
        self.model_type = config.get('model_type', 'Absolute')
        self.bucket_size = params.get('bucket_size', 1.0)
        self.seq_len = params.get('seq_len', 3)
        self.all_vals = params.get('all_vals', [])
        self.all_changes = params.get('all_changes', [])
        self.abs_map = self._deserialize_map(params.get('abs_map', {}))
        self.der_map = self._deserialize_map(params.get('der_map', {}))

    def _deserialize_map(self, serialized_map: dict) -> dict:
        deserialized = {}
        for k, v in serialized_map.items():
            if k == "":
                tuple_key = ()
            else:
                try:
                    tuple_key = tuple(int(x) for x in k.split("|"))
                except ValueError:
                    continue 
            
            inner_counter = Counter()
            for inner_k, inner_v in v.items():
                try:
                    inner_counter[int(inner_k)] = inner_v
                except ValueError:
                    continue
            deserialized[tuple_key] = inner_counter
        return deserialized

    def _get_bucket(self, price: float) -> int:
        bs = self.bucket_size
        if bs <= 0: bs = 1e-9
        if price >= 0:
            return int(price // bs)
        else:
            return int(price // bs) - 1

    def get_prediction_value(self, recent_prices: List[float]) -> int:
        if len(recent_prices) < self.seq_len + 1:
            return self._get_bucket(recent_prices[-1]) if recent_prices else 0
            
        buckets = [self._get_bucket(p) for p in recent_prices]
        window = buckets[-(self.seq_len + 1):] 
        a_seq = tuple(window[1:]) 
        
        if self.seq_len > 1:
            d_seq = tuple(a_seq[k] - a_seq[k-1] for k in range(1, len(a_seq)))
        else:
            d_seq = ()
            
        last_val = window[-1]
        
        if self.model_type == "Absolute":
            if a_seq in self.abs_map:
                return self.abs_map[a_seq].most_common(1)[0][0]
            return getattr(self, 'all_vals', [last_val])[0] if self.all_vals else last_val
            
        elif self.model_type == "Derivative":
            if d_seq in self.der_map:
                change = self.der_map[d_seq].most_common(1)[0][0]
                return last_val + change
            change = getattr(self, 'all_changes', [0])[0] if self.all_changes else 0
            return last_val + change
            
        elif self.model_type == "Combined":
            abs_cand = self.abs_map.get(a_seq, Counter())
            der_cand = self.der_map.get(d_seq, Counter())
            poss = set(abs_cand.keys())
            for c in der_cand.keys(): poss.add(last_val + c)
            
            if not poss: 
                return getattr(self, 'all_vals', [last_val])[0] if self.all_vals else last_val
            
            best, max_s = last_val, -1
            for v in poss:
                s = abs_cand[v] + der_cand[v - last_val]
                if s > max_s: max_s, best = s, v
            return best
        return last_val

class EnsembleStrategy:
    def __init__(self, asset: str, timeframe: str, strategy_union: List[dict], listed_acc: float):
        self.asset = asset
        self.timeframe = timeframe
        self.id = f"{asset}_{timeframe}"
        self.listed_accuracy = listed_acc
        
        self.sub_strategies = []
        for s_data in strategy_union:
             if 'config' in s_data and 'trained_parameters' in s_data:
                 self.sub_strategies.append(SubStrategy(s_data))
        
        if self.sub_strategies:
            self.min_bucket_size = min(s.bucket_size for s in self.sub_strategies)
        else:
            self.min_bucket_size = 0.0

    def predict(self, recent_prices: List[float]) -> int:
        if not recent_prices or not self.sub_strategies: return 0
        votes = []
        for strat in self.sub_strategies:
            pred_bucket = strat.get_prediction_value(recent_prices)
            current_bucket = strat._get_bucket(recent_prices[-1])
            diff = pred_bucket - current_bucket
            if diff > 0: votes.append(1)
            elif diff < 0: votes.append(-1)
            else: votes.append(0)
            
        up = votes.count(1)
        down = votes.count(-1)
        if up > down: return 1
        elif down > up: return -1
        return 0

# --- Backtesting Engine ---

class Backtester:
    def __init__(self):
        self.strategies = []
        self.price_data = {}
        self.results = []

    def load_strategies(self):
        logger.info(f"Fetching strategies from {GITHUB_API_URL}...")
        try:
            resp = requests.get(GITHUB_API_URL)
            if resp.status_code != 200:
                logger.error(f"GitHub API Error: {resp.status_code}")
                return

            files = resp.json()
            for f in files:
                if f['name'].endswith(".json") and f['name'] != "performance.json":
                    try:
                        data = requests.get(f['download_url']).json()
                        asset = data.get('asset')
                        tf = data.get('timeframe')
                        acc = data.get('combined_accuracy', 0)
                        union = data.get('strategy_union', [])
                        
                        if asset and tf and union:
                            strat = EnsembleStrategy(asset, tf, union, acc)
                            self.strategies.append(strat)
                            logger.info(f"Loaded {strat.id} (Listed Acc: {acc}%)")
                    except Exception as e:
                        logger.error(f"Error loading {f['name']}: {e}")
        except Exception as e:
            logger.error(f"Failed to load strategies: {e}")

    def fetch_binance_data(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Fetches historical klines from Binance handling pagination."""
        end_time = int(time.time() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        all_klines = []
        current_start = start_time
        
        logger.info(f"Fetching {days} days of {interval} data for {symbol}...")
        
        while True:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_time,
                "limit": 1000
            }
            try:
                resp = requests.get(url, params=params)
                data = resp.json()
                
                if not isinstance(data, list) or len(data) == 0:
                    break
                    
                all_klines.extend(data)
                last_ts = data[-1][0]
                
                if last_ts >= end_time or len(data) < 1000:
                    break
                
                current_start = last_ts + 1
                time.sleep(0.1) # Respect rate limits
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                break

        if not all_klines:
            return pd.DataFrame()

        df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
        ])
        df['close'] = df['close'].astype(float)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        return df[['open_time', 'close']]

    def resample_data(self, df: pd.DataFrame, target_tf: str) -> List[float]:
        """Resamples 15m data to target timeframe if needed."""
        if target_tf == "15m":
            return df['close'].tolist()
        
        tf_map = {"30m": "30min", "60m": "1h", "240m": "4h", "1d": "1D"}
        target = tf_map.get(target_tf)
        if not target:
            return df['close'].tolist()

        df_res = df.set_index('open_time')
        df_res = df_res['close'].resample(target).last().dropna()
        return df_res.tolist()

    def run_backtest(self):
        # 1. Gather unique assets to fetch
        unique_assets = set(s.asset for s in self.strategies)
        base_data = {}
        
        # 2. Fetch Base Data (15m)
        for asset in unique_assets:
            df = self.fetch_binance_data(asset, "15m", BACKTEST_DAYS)
            if not df.empty:
                base_data[asset] = df
            else:
                logger.warning(f"No data found for {asset}")

        print("\n" + "="*80)
        print(f"{'STRATEGY ID':<20} | {'LISTED %':<10} | {'BACKTEST %':<10} | {'TRADES':<8} | {'ACT. RATIO':<10} | {'STATUS'}")
        print("="*80)

        # 3. Simulate per Strategy
        for strat in self.strategies:
            if strat.asset not in base_data: continue
            
            # Prepare data
            raw_prices = self.resample_data(base_data[strat.asset], strat.timeframe)
            if len(raw_prices) < 50: continue

            wins = 0
            losses = 0
            flats_outcome = 0
            
            # Sliding window simulation
            # Need roughly max(seq_len) history for first prediction. 
            # Assuming max seq_len is around 10 to be safe.
            warmup = 20 
            
            for i in range(warmup, len(raw_prices) - 1):
                # Context Window
                context = raw_prices[i-warmup : i+1]
                current_price = raw_prices[i]
                next_price = raw_prices[i+1]
                
                # Predict
                signal = strat.predict(context)
                
                # Evaluate (Logic from PerformanceTracker)
                if signal == 0:
                    continue # No trade taken

                price_diff = next_price - current_price
                threshold = strat.min_bucket_size
                
                # Check for Directional Move (Non-flat outcome)
                if abs(price_diff) < threshold:
                    flats_outcome += 1
                    continue # Outcome was flat, prediction discarded
                
                is_win = False
                if signal == 1 and price_diff > 0: is_win = True
                elif signal == -1 and price_diff < 0: is_win = True
                
                if is_win:
                    wins += 1
                else:
                    losses += 1
            
            # Metrics
            total_trades = wins + losses
            if total_trades > 0:
                dma = (wins / total_trades) * 100
                activity_ratio = total_trades / (total_trades + flats_outcome)
            else:
                dma = 0.0
                activity_ratio = 0.0
                
            # deviation check
            deviation = dma - strat.listed_accuracy
            status = "OK"
            if abs(deviation) > 20.0:
                status = "!!! ALARM !!!"
            elif total_trades < 10:
                status = "LOW DATA"
                
            print(f"{strat.id:<20} | {strat.listed_accuracy:>8.2f}% | {dma:>8.2f}%   | {total_trades:>8} | {activity_ratio:>8.2f}   | {status}")

if __name__ == "__main__":
    bt = Backtester()
    bt.load_strategies()
    bt.run_backtest()
