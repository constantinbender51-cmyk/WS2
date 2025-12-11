import ccxt
import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template_string
import io
import base64
import copy
from collections import Counter
import threading
import time

app = Flask(__name__)

# --- CONFIGURATION ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE = '2018-01-01 00:00:00'

# GA Settings
POPULATION_SIZE = 300   
GENERATIONS = 50        
MUTATION_RATE = 0.2    
TOURNAMENT_SIZE = 5     
ELITISM_COUNT = 10      

# Ensemble Settings
ANCHOR_STEP = 400 
TOP_N_STRATEGIES = 50   

# Diversity Settings
# Strategies must have distinct iii_windows.
# Since iii_window is int, any diff < 1 means they are identical.
DIVERSITY_THRESHOLD = 0.5 

# Parameter Constraints
GENE_SPACE = {
    'sma_fast':   (10, 200, int),
    'sma_slow':   (50, 400, int),  
    'sl_pct':     (0.01, 0.15, float),
    'tp_pct':     (0.05, 0.50, float),
    'iii_window': (10, 90, int),
    't_low':      (0.05, 0.30, float),
    't_high':     (0.20, 0.60, float), 
    'lev_1':      (0.0, 5.0, float),   
    'lev_2':      (0.0, 5.0, float),   
    'lev_3':      (0.0, 5.0, float)    
}

# --- GLOBAL STATE ---
# Stores the results to be accessed by the web server
GLOBAL_RESULTS = None
# Stores the current status string for display
ANALYSIS_STATUS = "Initializing..."

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
        
        # Fitness: Sharpe
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

def calculate_strategy_similarity(strat1, strat2):
    """
    STRICT SIMILARITY: Calculates difference in iii_window ONLY.
    """
    # Simply return the absolute difference in window size.
    # 0 means identical window.
    return abs(strat1['iii_window'] - strat2['iii_window'])

def select_diverse_strategies(scored_population, n, threshold):
    """
    Selects top N strategies.
    Rejects any strategy whose iii_window is the same as (or within threshold of)
    any already selected strategy.
    """
    selected = []
    
    # 1. Always take the absolute best
    if not scored_population:
        return []
        
    selected.append(scored_population[0])
    
    # 2. Try to fill with distinct iii_windows
    for i in range(1, len(scored_population)):
        candidate = scored_population[i]
        is_distinct = True
        
        for picked in selected:
            # Check similarity (difference in iii_window)
            diff = calculate_strategy_similarity(candidate[0], picked[0])
            if diff < threshold: # If identical iii_window
                is_distinct = False
                break
        
        if is_distinct:
            selected.append(candidate)
        
        if len(selected) >= n:
            break
            
    # 3. Fallback: If we filtered too aggressively (ran out of unique windows)
    # Fill with next best unique strategies even if windows duplicate (unlikely given range)
    if len(selected) < n:
        # Just grab unique param sets to fill quota
        selected_strs = {str(s[0]) for s in selected}
        for item in scored_population:
            if str(item[0]) not in selected_strs:
                selected.append(item)
                selected_strs.add(str(item[0]))
                if len(selected) >= n:
                    break
                    
    return selected

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

    # Final Evaluation & Diversity Selection
    final_scored = []
    for ind in population:
        if ind['sma_slow'] <= ind['sma_fast'] or ind['t_high'] <= ind['t_low']:
            fitness = -10.0
        else:
            fitness = evaluator.run_backtest(window_df, ind)
        final_scored.append((ind, fitness))
    
    final_scored.sort(key=lambda x: x[1], reverse=True)
    
    # Use the III-Based Diversity Filter
    diverse_top_n = select_diverse_strategies(final_scored, TOP_N_STRATEGIES, DIVERSITY_THRESHOLD)
    
    return diverse_top_n

# --- BACKGROUND WORKER ---

def background_ga_worker():
    """Runs independently of the web server"""
    global GLOBAL_RESULTS, ANALYSIS_STATUS
    
    print("GA WORKER: Starting...")
    ANALYSIS_STATUS = "Fetching Data..."
    
    try:
        df = fetch_data()
        
        ANALYSIS_STATUS = "Running Anchored GA Analysis (This takes time)..."
        evaluator = StrategyEvaluator(df)
        results = {}
        
        max_days = len(df)
        total_steps = len(range(ANCHOR_STEP, max_days + 1, ANCHOR_STEP))
        current_step = 0
        
        for end_day in range(ANCHOR_STEP, max_days + 1, ANCHOR_STEP):
            current_step += 1
            ANALYSIS_STATUS = f"Processing Window {current_step}/{total_steps}: First {end_day} days..."
            print(f"GA WORKER: {ANALYSIS_STATUS}")
            
            window_df = df.iloc[:end_day]
            top_strategies = run_ga_for_window(evaluator, window_df)
            results[end_day] = top_strategies
            
        ANALYSIS_STATUS = "Generating Plots & Consistency Scores..."
        print("GA WORKER: Finalizing results...")
        
        processed_results = process_results_for_display(results)
        plot_data = generate_ensemble_plot(results)
        
        GLOBAL_RESULTS = {
            'processed_results': processed_results,
            'plot_data': plot_data
        }
        
        ANALYSIS_STATUS = "Complete"
        print("GA WORKER: Analysis Complete.")
        
    except Exception as e:
        ANALYSIS_STATUS = f"Error: {str(e)}"
        print(f"GA WORKER ERROR: {e}")
        import traceback
        traceback.print_exc()

def process_results_for_display(results):
    """
    Augments results with GLOBAL CONSISTENCY scores based on III Window stability.
    """
    sorted_days = sorted(results.keys())
    
    all_scores = []
    augmented_results = {} 
    for day in sorted_days:
        augmented_results[day] = []
    
    for current_day in sorted_days:
        current_strategies = results[current_day]
        
        for idx, (strat, fitness) in enumerate(current_strategies):
            
            total_similarity_diff = 0.0
            comparison_count = 0
            
            for other_day in sorted_days:
                if other_day == current_day:
                    continue
                
                other_strategies = results[other_day]
                
                # Find best match in OTHER window (Closest III Window)
                best_match_diff = float('inf')
                for other_strat, _ in other_strategies:
                    # diff is now just the integer difference in iii_window
                    diff = calculate_strategy_similarity(strat, other_strat)
                    if diff < best_match_diff:
                        best_match_diff = diff
                
                total_similarity_diff += best_match_diff
                comparison_count += 1
            
            if comparison_count > 0:
                avg_consistency = total_similarity_diff / comparison_count
            else:
                avg_consistency = 999.0
            
            augmented_results[current_day].append({
                'strat': strat,
                'fitness': fitness,
                'similarity': avg_consistency,
                'is_stable': False
            })
            
            all_scores.append((current_day, idx, avg_consistency))
        
    if all_scores:
        all_scores.sort(key=lambda x: x[2]) 
        top_10_stable = all_scores[:10]
        
        for day, idx, score in top_10_stable:
            augmented_results[day][idx]['is_stable'] = True
        
    return augmented_results

def generate_ensemble_plot(results):
    fig, axes = plt.subplots(4, 1, figsize=(14, 20), sharex=True)
    
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
            avg_lev = (strat['lev_1'] + strat['lev_2'] + strat['lev_3']) / 3
            lev_avg_vals.append(avg_lev)
            sharpe_vals.append(fitness)

    axes[0].scatter(x_vals, sma_slow_vals, c='blue', alpha=0.5, label='SMA Slow', s=25)
    axes[0].scatter(x_vals, sma_fast_vals, c='cyan', alpha=0.5, label='SMA Fast', s=25)
    axes[0].set_title(f'Parameter Stability: Trend Definition (Top {TOP_N_STRATEGIES} Diverse)', fontsize=14)
    axes[0].set_ylabel('Period Length')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(x_vals, iii_vals, c='magenta', alpha=0.5, s=25)
    axes[1].set_title('III Window Diversity (Strict Separation Enforced)', fontsize=14)
    axes[1].set_ylabel('Lookback Period')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].scatter(x_vals, lev_avg_vals, c='green', alpha=0.5, s=25)
    axes[2].set_title('Strategy Aggressiveness (Avg Leverage Setting)', fontsize=14)
    axes[2].set_ylabel('Avg Leverage (0-5x)')
    axes[2].grid(True, alpha=0.3)

    axes[3].scatter(x_vals, sharpe_vals, c='gold', edgecolors='black', alpha=0.7, s=40)
    axes[3].set_title('In-Sample Performance (Sharpe) of Diverse Strategies', fontsize=14)
    axes[3].set_ylabel('Sharpe Ratio')
    axes[3].set_xlabel('Training Window Size (Days)')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    plt.close(fig)
    return data

@app.route('/')
def index():
    if GLOBAL_RESULTS is None:
        return render_template_string(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Analysis Running</title>
            <meta http-equiv="refresh" content="5">
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; text-align: center; padding-top: 50px; background: #f4f4f4; }}
                .loader {{ border: 16px solid #f3f3f3; border-top: 16px solid #3498db; border-radius: 50%; width: 120px; height: 120px; animation: spin 2s linear infinite; margin: 0 auto; }}
                @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
            </style>
        </head>
        <body>
            <h1>Genetic Algorithm Running</h1>
            <div class="loader"></div>
            <p><strong>Status:</strong> {ANALYSIS_STATUS}</p>
            <p>Population: {POPULATION_SIZE} | Generations: {GENERATIONS}</p>
            <p>The page will reload automatically when results are ready.</p>
        </body>
        </html>
        """)

    processed_results = GLOBAL_RESULTS['processed_results']
    plot_data = GLOBAL_RESULTS['plot_data']
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ensemble GA Strategy Report</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
            .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-radius: 8px; }}
            h1, h2 {{ color: #2c3e50; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin-bottom: 30px; border-radius: 4px; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; font-size: 0.85em; }}
            th, td {{ border: 1px solid #e0e0e0; padding: 10px; text-align: center; }}
            th {{ background-color: #f8f9fa; font-weight: 600; color: #555; }}
            tr:nth-child(even) {{ background-color: #fafafa; }}
            
            tr.stable-green td {{ background-color: #d4edda !important; color: #155724; border-color: #c3e6cb; }}
            tr.stable-green:hover td {{ background-color: #c3e6cb !important; }}

            .window-header {{ background-color: #34495e; color: white; padding: 12px; margin-top: 40px; border-radius: 5px 5px 0 0; font-weight: bold; }}
            .similarity-note {{ font-size: 0.8em; color: #777; margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Anchored Ensemble GA Analysis</h1>
            <p>Analysis of top {TOP_N_STRATEGIES} strategies evolved over growing time windows.</p>
            <p><strong>Config:</strong> Pop={POPULATION_SIZE}, Gens={GENERATIONS}, Step={ANCHOR_STEP}d</p>
            <p><strong>Diversity Rule:</strong> Strategies must have EXACTLY distinct III Windows.</p>
            
            <h2>Parameter Stability Visualization</h2>
            <img src="data:image/png;base64,{plot_data}" alt="Ensemble Plot">
            
            <h2>Detailed Strategy Parameters</h2>
            <p class="similarity-note">
                <strong>III Consistency Score:</strong> Measures how consistently this specific III Window appeared in top strategies across OTHER historical eras.
                <br>Score is the average difference in III Window size compared to best matches in other eras. 
                <br>Lower score = This window size is a "Universal Performer".
                Rows highlighted in <strong>GREEN</strong> are the most historically stable window sizes.
            </p>
    """
    
    for days in sorted(processed_results.keys()):
        strategies = processed_results[days]
        html_content += f"""
        <div class="window-header">Training Window: First {days} Days</div>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Sharpe</th>
                    <th>III Consistency</th>
                    <th>III Window</th>
                    <th>SMA Fast</th>
                    <th>SMA Slow</th>
                    <th>T Low</th>
                    <th>T High</th>
                    <th>Lev 1 (Low)</th>
                    <th>Lev 2 (Mid)</th>
                    <th>Lev 3 (High)</th>
                    <th>SL %</th>
                    <th>TP %</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for i, item in enumerate(strategies):
            strat = item['strat']
            fitness = item['fitness']
            sim_score = item['similarity']
            is_stable = item['is_stable']
            
            row_class = "stable-green" if is_stable else ""
            sim_display = f"{sim_score:.2f}" if sim_score < 900 else "N/A"
            
            html_content += f"""
                <tr class="{row_class}">
                    <td>{i+1}</td>
                    <td>{fitness:.4f}</td>
                    <td>{sim_display}</td>
                    <td>{strat['iii_window']}</td>
                    <td>{strat['sma_fast']}</td>
                    <td>{strat['sma_slow']}</td>
                    <td>{strat['t_low']:.2f}</td>
                    <td>{strat['t_high']:.2f}</td>
                    <td>{strat['lev_1']:.1f}</td>
                    <td>{strat['lev_2']:.1f}</td>
                    <td>{strat['lev_3']:.1f}</td>
                    <td>{strat['sl_pct']:.2f}</td>
                    <td>{strat['tp_pct']:.2f}</td>
                </tr>
            """
            
        html_content += """
            </tbody>
        </table>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    return render_template_string(html_content)

if __name__ == '__main__':
    # Start GA in background thread
    t = threading.Thread(target=background_ga_worker)
    t.daemon = True
    t.start()
    
    print("Starting Web Server (Analysis running in background)...")
    app.run(host='0.0.0.0', port=8080)
