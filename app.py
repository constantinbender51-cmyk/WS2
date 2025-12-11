import ccxt
import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, send_file
import io
import copy
from collections import Counter

app = Flask(__name__)

# --- CONFIGURATION ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE = '2018-01-01 00:00:00'

# GA Settings (Doubled)
POPULATION_SIZE = 100  
GENERATIONS = 30       
MUTATION_RATE = 0.2    
TOURNAMENT_SIZE = 4    
ELITISM_COUNT = 5      

# Ensemble Settings
ANCHOR_STEP = 400 # Days to add in each training iteration
TOP_N_STRATEGIES = 10 # Strategies to keep per window

# Parameter Constraints (Expanded Leverage)
GENE_SPACE = {
    'sma_fast':   (10, 200, int),
    'sma_slow':   (50, 400, int),  
    'sl_pct':     (0.01, 0.15, float),
    'tp_pct':     (0.05, 0.50, float),
    'iii_window': (10, 90, int),
    't_low':      (0.05, 0.30, float),
    't_high':     (0.20, 0.60, float), 
    'lev_1':      (0.0, 5.0, float),   # Expanded 0-5
    'lev_2':      (0.0, 5.0, float),   
    'lev_3':      (0.0, 5.0, float)    
}

def fetch_data():
    """Fetches OHLCV data from Binance"""
    print(f"Fetching {SYMBOL} data...")
    exchange = ccxt.binance()
    since = exchange.parse8601(START_DATE)
    all_candles = []
    
    while True:
        try:
            candles = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
            if not candles: break
            since = candles[-1][0] + 1
            all_candles += candles
            if len(candles) < 1000: break
        except Exception as e:
            print(f"Fetch Error: {e}")
            break
            
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

class StrategyEvaluator:
    def __init__(self, df):
        self.df = df
        
    def calculate_indicators(self, df_in, params):
        df = df_in.copy()
        # Calculate SMAs
        df['sma_fast'] = df['close'].rolling(int(params['sma_fast'])).mean()
        df['sma_slow'] = df['close'].rolling(int(params['sma_slow'])).mean()
        
        # Calculate III (Efficiency Ratio)
        period = int(params['iii_window'])
        log_ret = np.log(df['close'] / df['close'].shift(1))
        numerator = log_ret.rolling(period).sum().abs()
        denominator = log_ret.abs().rolling(period).sum()
        df['iii'] = numerator / (denominator + 1e-9)
        
        return df

    def run_backtest(self, df_in, params):
        """Vector-optimized backtest for speed during GA"""
        df = self.calculate_indicators(df_in, params)
        
        # Pre-calculation
        closes = df['close'].values
        sma_fast = df['sma_fast'].values
        sma_slow = df['sma_slow'].values
        iii = df['iii'].values
        
        # Logic Vectors
        # 1. Determine Trend
        long_signal = (closes > sma_fast) & (closes > sma_slow)
        short_signal = (closes < sma_fast) & (closes < sma_slow)
        
        # 2. Determine Leverage
        t_low, t_high = params['t_low'], params['t_high']
        lev_1, lev_2, lev_3 = params['lev_1'], params['lev_2'], params['lev_3']
        
        lev_vector = np.full(len(df), lev_3) # Default High
        lev_vector[iii < t_high] = lev_2     # Mid
        lev_vector[iii < t_low] = lev_1      # Low
        
        # 3. Calculate Strategy Returns
        market_ret = np.diff(np.log(closes), prepend=0)
        
        # Position Vector (Shifted by 1 to avoid lookahead)
        # Position today is determined by signal yesterday
        pos_vector = np.zeros(len(df))
        pos_vector[long_signal] = 1
        pos_vector[short_signal] = -1
        pos_vector = np.roll(pos_vector, 1)
        pos_vector[0] = 0
        
        # Leverage Vector (Shifted by 1)
        lev_vector = np.roll(lev_vector, 1)
        lev_vector[0] = 0
        
        # Final Returns
        strategy_ret = pos_vector * lev_vector * market_ret
        
        # Simple Equity (for fitness check, we use Log Returns sum)
        # We ignore SL/TP in the GA search for pure speed, 
        # relying on the volatility (Sharpe) to naturally penalize bad entries.
        total_log_ret = np.sum(strategy_ret)
        std_dev = np.std(strategy_ret)
        
        if std_dev == 0: return -10.0
        sharpe = np.sqrt(365) * (np.mean(strategy_ret) / std_dev)
        
        return sharpe

# --- GENETIC ALGORITHM ENGINE ---

def random_gene(name):
    min_v, max_v, dtype = GENE_SPACE[name]
    if dtype == int:
        return random.randint(min_v, max_v)
    else:
        return random.uniform(min_v, max_v)

def create_individual():
    return {k: random_gene(k) for k in GENE_SPACE.keys()}

def mutate(individual):
    ind = individual.copy()
    gene_key = random.choice(list(GENE_SPACE.keys()))
    ind[gene_key] = random_gene(gene_key)
    return ind

def crossover(p1, p2):
    child = {}
    for k in GENE_SPACE.keys():
        child[k] = p1[k] if random.random() > 0.5 else p2[k]
    return child

def run_ga_for_window(evaluator, window_df):
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    
    for gen in range(GENERATIONS):
        scored_pop = []
        for ind in population:
            # Constraints
            if ind['sma_slow'] <= ind['sma_fast']: 
                fitness = -10.0
            elif ind['t_high'] <= ind['t_low']: 
                fitness = -10.0
            else:
                fitness = evaluator.run_backtest(window_df, ind)
            scored_pop.append((ind, fitness))
        
        scored_pop.sort(key=lambda x: x[1], reverse=True)
        
        # Next Gen
        next_gen = [x[0] for x in scored_pop[:ELITISM_COUNT]]
        
        while len(next_gen) < POPULATION_SIZE:
            parents = random.sample(scored_pop[:50], 2) # Top 50 tournament
            child = crossover(parents[0][0], parents[1][0])
            if random.random() < MUTATION_RATE:
                child = mutate(child)
            next_gen.append(child)
        population = next_gen

    # Final Evaluation
    final_scored = []
    for ind in population:
        if ind['sma_slow'] <= ind['sma_fast'] or ind['t_high'] <= ind['t_low']:
            fitness = -10.0
        else:
            fitness = evaluator.run_backtest(window_df, ind)
        final_scored.append((ind, fitness))
    
    final_scored.sort(key=lambda x: x[1], reverse=True)
    return final_scored[:TOP_N_STRATEGIES]

# --- MAIN EXECUTION ---

def run_anchored_analysis(df):
    evaluator = StrategyEvaluator(df)
    results = {}
    
    # Loop from 400 days to end of data
    max_days = len(df)
    
    print(f"Starting Anchored Ensemble Analysis on {max_days} days of data...")
    
    for end_day in range(ANCHOR_STEP, max_days + 1, ANCHOR_STEP):
        window_df = df.iloc[:end_day]
        print(f"Training on first {end_day} days...")
        
        top_strategies = run_ga_for_window(evaluator, window_df)
        results[end_day] = top_strategies
        
    return results

def generate_ensemble_plot(results):
    fig, axes = plt.subplots(4, 1, figsize=(14, 20), sharex=True)
    
    # Extract data for plotting
    x_vals = []
    sma_fast_vals = []
    sma_slow_vals = []
    iii_vals = []
    lev_avg_vals = []
    sharpe_vals = []
    
    for days, strategies in results.items():
        for strat, fitness in strategies:
            x_vals.append(days)
            sma_fast_vals.append(strat['sma_fast'])
            sma_slow_vals.append(strat['sma_slow'])
            iii_vals.append(strat['iii_window'])
            # Average leverage aggressiveness
            avg_lev = (strat['lev_1'] + strat['lev_2'] + strat['lev_3']) / 3
            lev_avg_vals.append(avg_lev)
            sharpe_vals.append(fitness)

    # 1. SMA Stability
    axes[0].scatter(x_vals, sma_slow_vals, c='blue', alpha=0.6, label='SMA Slow', s=30)
    axes[0].scatter(x_vals, sma_fast_vals, c='cyan', alpha=0.6, label='SMA Fast', s=30)
    axes[0].set_title('Parameter Stability: Trend Definition (SMA)', fontsize=14)
    axes[0].set_ylabel('Period Length')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. III Window Stability
    axes[1].scatter(x_vals, iii_vals, c='magenta', alpha=0.6, s=30)
    axes[1].set_title('Parameter Stability: Efficiency Window (III)', fontsize=14)
    axes[1].set_ylabel('Lookback Period')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Leverage Aggressiveness
    axes[2].scatter(x_vals, lev_avg_vals, c='green', alpha=0.6, s=30)
    axes[2].set_title('Strategy Aggressiveness (Avg Leverage Setting)', fontsize=14)
    axes[2].set_ylabel('Avg Leverage (0-5x)')
    axes[2].grid(True, alpha=0.3)

    # 4. Performance Degradation/Improvement
    axes[3].scatter(x_vals, sharpe_vals, c='gold', edgecolors='black', alpha=0.8, s=40)
    axes[3].set_title('In-Sample Performance (Sharpe) of Top 10 Strategies', fontsize=14)
    axes[3].set_ylabel('Sharpe Ratio')
    axes[3].set_xlabel('Training Window Size (Days)')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    return buf

@app.route('/')
def index():
    df = fetch_data()
    results = run_anchored_analysis(df)
    img_buf = generate_ensemble_plot(results)
    return send_file(img_buf, mimetype='image/png')

if __name__ == '__main__':
    print("Starting Ensemble GA Server...")
    app.run(host='0.0.0.0', port=8080)
