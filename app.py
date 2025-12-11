import ccxt
import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, send_file
import io
import copy

app = Flask(__name__)

# --- CONFIGURATION ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE = '2018-01-01 00:00:00'

# GA Settings
POPULATION_SIZE = 100   # Number of strategies in each generation
GENERATIONS = 30       # Number of evolution cycles
MUTATION_RATE = 0.2    # Probability of a gene mutating
TOURNAMENT_SIZE = 3    # Selection pressure
ELITISM_COUNT = 2      # Keep top N strategies unchanged

# Parameter Constraints (Min, Max, Step/Type)
GENE_SPACE = {
    'sma_fast':   (10, 200, int),
    'sma_slow':   (50, 400, int),  # Constraint: Slow > Fast checked in logic
    'sl_pct':     (0.01, 0.15, float),
    'tp_pct':     (0.05, 0.50, float),
    'iii_window': (10, 60, int),
    't_low':      (0.05, 0.30, float),
    't_high':     (0.20, 0.60, float), # Constraint: High > Low checked in logic
    'lev_1':      (0.1, 5, float),   # Low Volatility / Bad regime
    'lev_2':      (0.1, 5, float),   # Mid
    'lev_3':      (0.1, 5.0, float)    # High Conviction
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
        self.train_idx = int(len(df) * 0.5)
        self.train_data = df.iloc[:self.train_idx].copy()
        
    def calculate_indicators(self, df, params):
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
        """
        Runs the logic row-by-row for accurate SL/TP handling.
        Returns: Equity Curve (Series)
        """
        df = df_in.copy()
        df = self.calculate_indicators(df, params)
        
        # Extract numpy arrays for speed
        closes = df['close'].values
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        sma_fast = df['sma_fast'].values
        sma_slow = df['sma_slow'].values
        iii = df['iii'].values # Note: We use Shifted III in logic usually
        
        n = len(df)
        equity = np.zeros(n)
        equity[0] = 1.0
        current_equity = 1.0
        
        # Parameter Unpacking
        sl_pct = params['sl_pct']
        tp_pct = params['tp_pct']
        t_low = params['t_low']
        t_high = params['t_high']
        lev_map = {0: params['lev_1'], 1: params['lev_2'], 2: params['lev_3']}
        
        # Start after longest lookback
        start_idx = max(int(params['sma_slow']), int(params['iii_window'])) + 1
        
        for i in range(1, n):
            if i < start_idx:
                equity[i] = current_equity
                continue
                
            # Logic uses PREVIOUS candle to decide entry
            prev_c = closes[i-1]
            prev_fast = sma_fast[i-1]
            prev_slow = sma_slow[i-1]
            prev_iii = iii[i-1]
            
            # Determine Leverage based on III
            if prev_iii < t_low: lev_tier = 0
            elif prev_iii < t_high: lev_tier = 1
            else: lev_tier = 2
            leverage = lev_map[lev_tier]
            
            # Trend Check
            signal = 0 # 1 Long, -1 Short
            if prev_c > prev_fast and prev_c > prev_slow:
                signal = 1
            elif prev_c < prev_fast and prev_c < prev_slow:
                signal = -1
            
            # Calculate PnL for this candle
            step_ret = 0.0
            
            if signal == 1:
                entry = opens[i]
                stop_loss = entry * (1 - sl_pct)
                take_profit = entry * (1 + tp_pct)
                
                # Check Low/High for SL/TP
                if lows[i] <= stop_loss:
                    step_ret = -sl_pct * leverage
                elif highs[i] >= take_profit:
                    step_ret = tp_pct * leverage
                else:
                    step_ret = ((closes[i] - entry) / entry) * leverage
                    
            elif signal == -1:
                entry = opens[i]
                stop_loss = entry * (1 + sl_pct)
                take_profit = entry * (1 - tp_pct)
                
                if highs[i] >= stop_loss:
                    step_ret = -sl_pct * leverage
                elif lows[i] <= take_profit:
                    step_ret = tp_pct * leverage
                else:
                    step_ret = ((entry - closes[i]) / entry) * leverage
            
            # Update Equity
            current_equity *= (1 + step_ret)
            equity[i] = current_equity
            
        return pd.Series(equity, index=df.index)

    def evaluate_fitness(self, params):
        """Calculates Sharpe Ratio on TRAINING data"""
        # Constraint Checks
        if params['sma_slow'] <= params['sma_fast']: return -10.0
        if params['t_high'] <= params['t_low']: return -10.0
        
        equity = self.run_backtest(self.train_data, params)
        
        # Calculate Sharpe
        returns = equity.pct_change().dropna()
        if returns.std() == 0: return -10.0
        sharpe = np.sqrt(365) * (returns.mean() / returns.std())
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
    # Uniform crossover
    child = {}
    for k in GENE_SPACE.keys():
        child[k] = p1[k] if random.random() > 0.5 else p2[k]
    return child

def run_genetic_algorithm(evaluator):
    # 1. Initialize
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    print(f"Genetic Algorithm Started: {POPULATION_SIZE} individuals, {GENERATIONS} generations")
    
    for gen in range(GENERATIONS):
        # 2. Evaluate
        scored_pop = []
        for ind in population:
            fitness = evaluator.evaluate_fitness(ind)
            scored_pop.append((ind, fitness))
        
        # Sort by fitness (descending)
        scored_pop.sort(key=lambda x: x[1], reverse=True)
        
        best_fitness = scored_pop[0][1]
        print(f"Generation {gen+1}/{GENERATIONS} | Best Train Sharpe: {best_fitness:.4f}")
        
        # 3. Selection & Next Gen
        next_gen = []
        
        # Elitism
        for i in range(ELITISM_COUNT):
            next_gen.append(scored_pop[i][0])
            
        # Breeding
        while len(next_gen) < POPULATION_SIZE:
            # Tournament Selection
            parents = random.sample(scored_pop[:len(scored_pop)//2], 2) # Sample from top 50%
            parent1 = parents[0][0]
            parent2 = parents[1][0]
            
            child = crossover(parent1, parent2)
            
            # Mutation
            if random.random() < MUTATION_RATE:
                child = mutate(child)
                
            next_gen.append(child)
            
        population = next_gen
        
    # Return best individual from final generation
    final_scored = [(ind, evaluator.evaluate_fitness(ind)) for ind in population]
    final_scored.sort(key=lambda x: x[1], reverse=True)
    return final_scored[0][0]

# --- PLOTTING ---
def generate_plot(df, best_equity, train_cutoff, best_params):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # 1. Equity Curve
    ax1.plot(best_equity.index, best_equity, color='blue', label='AI Optimized Strategy', linewidth=2)
    
    # Buy & Hold (Benchmark)
    bnh = df['close'] / df['close'].iloc[0]
    ax1.plot(bnh.index, bnh, color='gray', linestyle='--', alpha=0.5, label='Buy & Hold')
    
    # The Wall of Truth
    ax1.axvline(train_cutoff, color='red', linewidth=3, label='End of Training Data')
    ax1.axvspan(train_cutoff, df.index[-1], color='red', alpha=0.05, label='Out-of-Sample (Test)')
    
    ax1.set_title(f'Genetic Algo Stress Test: {SYMBOL}', fontsize=14)
    ax1.set_ylabel('Equity (Log Scale)')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Text Box with Parameters
    param_str = "Best Genomes Found:\n"
    for k, v in best_params.items():
        val = f"{v:.4f}" if isinstance(v, float) else f"{v}"
        param_str += f"{k}: {val}\n"
        
    ax1.text(0.02, 0.95, param_str, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # 2. Drawdown
    dd = best_equity / best_equity.cummax() - 1
    ax2.fill_between(dd.index, dd, 0, color='red', alpha=0.3)
    ax2.set_ylabel('Drawdown')
    ax2.set_title('Drawdown Profile')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

@app.route('/')
def index():
    # 1. Fetch
    df = fetch_data()
    
    # 2. Setup Evaluator
    evaluator = StrategyEvaluator(df)
    train_cutoff = df.index[evaluator.train_idx]
    
    # 3. Run Genetic Algorithm
    best_params = run_genetic_algorithm(evaluator)
    
    # 4. Run Full Backtest (Train + Test)
    full_equity = evaluator.run_backtest(df, best_params)
    
    # 5. Calculate Performance
    # Train Stats
    train_eq = full_equity.loc[:train_cutoff]
    train_ret = train_eq.iloc[-1]
    
    # Test Stats
    test_eq = full_equity.loc[train_cutoff:]
    test_ret = test_eq.iloc[-1] / test_eq.iloc[0]
    
    img_buf = generate_plot(df, full_equity, train_cutoff, best_params)
    
    return send_file(img_buf, mimetype='image/png')

if __name__ == '__main__':
    print("Starting Genetic Optimizer Server...")
    app.run(host='0.0.0.0', port=8080)
