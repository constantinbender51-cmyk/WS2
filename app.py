import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string, jsonify
import random
from datetime import datetime
import time
import threading
from functools import lru_cache

app = Flask(__name__)

# --- CONFIGURATION ---
SYMBOL = 'BTCUSDT'
START_YEAR = 2018

# GA CONFIGURATION
POPULATION_SIZE = 60
GENERATIONS = 25
MUTATION_RATE = 0.15
TOURNAMENT_SIZE = 3
ELITISM_COUNT = 3

# PARAMETER BOUNDS
FAST_MIN, FAST_MAX = 5, 100
SLOW_RATIO_MIN, SLOW_RATIO_MAX = 1.1, 4.0   # Slow = Fast * Ratio (>1 to ensure Slow > Fast)
SIG_RATIO_MIN, SIG_RATIO_MAX = 0.2, 1.5     # Signal = Fast * Ratio (Standard is ~0.75)
WEIGHT_MIN, WEIGHT_MAX = 0.0, 1.0

# --- GLOBAL STATE ---
GLOBAL_STATE = {
    'status': 'Waiting to start...',
    'progress': 0,
    'done': False,
    'results': [],
    'plot': None,
    'error': None
}

# --- DATA FETCHING ---
def fetch_binance_data(symbol, interval, start_year):
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime(start_year, 1, 1).timestamp() * 1000)
    end_ts = int(time.time() * 1000)
    limit = 1000
    
    all_data = []
    current_start = start_ts
    
    req_count = 0
    while current_start < end_ts and req_count < 1000:
        params = {'symbol': symbol, 'interval': interval, 'startTime': current_start, 'limit': limit}
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            if not data: break
            all_data.extend(data)
            current_start = data[-1][0] + 1
            req_count += 1
            time.sleep(0.05)
        except: break
            
    df = pd.DataFrame(all_data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'ct', 'qav', 'nt', 'tbv', 'tqv', 'i'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close'] = df['close'].astype(float)
    df.set_index('open_time', inplace=True)
    return df[['close']]

# --- STRATEGY LOGIC & MEMOIZATION ---

# Cache key: (Fast, Slow, Signal) -> np.array
SIGNAL_CACHE = {}

def get_cached_signal(prices_tuple, fast, slow, signal_p):
    """
    Retrieves MACD signal from cache or calculates it.
    """
    key = (fast, slow, signal_p)
    if key in SIGNAL_CACHE:
        return SIGNAL_CACHE[key]

    prices = np.array(prices_tuple)
    series = pd.Series(prices)
    
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_p, adjust=False).mean()
    
    # +1 Long, -1 Short
    sig = np.where(macd_line > signal_line, 1.0, -1.0)
    sig = np.concatenate(([0], sig[:-1])) # Shift for lookahead bias
    
    SIGNAL_CACHE[key] = sig
    return sig

def calculate_sharpe_ratio(returns, periods_per_year):
    std = returns.std()
    if std == 0 or np.isnan(std): return -10.0
    return (returns.mean() / std) * np.sqrt(periods_per_year)

# --- GENETIC ALGORITHM ENGINE (12 VARIABLES) ---

def create_individual():
    """
    Creates a genome with 12 variables.
    Genes 0-2: Weights [W1, W2, W3]
    Genes 3-5: MACD 1 [Fast, SlowRatio, SigRatio]
    Genes 6-8: MACD 2 [Fast, SlowRatio, SigRatio]
    Genes 9-11: MACD 3 [Fast, SlowRatio, SigRatio]
    """
    genes = []
    # Weights
    genes.extend([random.uniform(WEIGHT_MIN, WEIGHT_MAX) for _ in range(3)])
    # MACD 1, 2, 3
    for _ in range(3):
        genes.append(random.randint(FAST_MIN, FAST_MAX))        # Fast
        genes.append(random.uniform(SLOW_RATIO_MIN, SLOW_RATIO_MAX)) # Slow Ratio
        genes.append(random.uniform(SIG_RATIO_MIN, SIG_RATIO_MAX))   # Signal Ratio
    return genes

def decode_individual(ind):
    """
    Decodes the 12-gene list into structured parameters.
    Returns: weights, macd_params [(fast, slow, sig), ...]
    """
    weights = ind[0:3]
    macd_genes = ind[3:] # Remaining 9 genes (3 sets of 3)
    
    params = []
    for i in range(0, 9, 3): # Step by 3: (Fast, SR, SigR)
        f_gene = macd_genes[i]
        sr_gene = macd_genes[i+1]
        sigr_gene = macd_genes[i+2]
        
        fast_p = int(f_gene)
        slow_p = int(fast_p * sr_gene)
        sig_p = int(fast_p * sigr_gene)
        
        # Constraints
        if slow_p <= fast_p: slow_p = fast_p + 1
        if sig_p < 1: sig_p = 1
        
        params.append((fast_p, slow_p, sig_p))
        
    return weights, params

def evaluate_fitness(individual, prices_tuple, returns, factor):
    weights, macd_params = decode_individual(individual)
    
    total_w = sum(weights)
    if total_w == 0: return -10.0
    
    # Get signals
    s1 = get_cached_signal(prices_tuple, *macd_params[0])
    s2 = get_cached_signal(prices_tuple, *macd_params[1])
    s3 = get_cached_signal(prices_tuple, *macd_params[2])
    
    # Composite Position
    pos = (weights[0]*s1 + weights[1]*s2 + weights[2]*s3) / total_w
    
    strat_rets = pos * returns
    return calculate_sharpe_ratio(strat_rets, factor)

def crossover(p1, p2):
    """Two-point crossover adapted for 12 genes"""
    if random.random() > 0.8: return p1[:], p2[:] 
    
    # Cut points between gene blocks to preserve MACD triplet integrity (mostly)
    # 0-2 (Weights), 3-5 (M1), 6-8 (M2), 9-11 (M3)
    # We choose cut points randomly within the full 12 range
    pt1 = random.randint(1, 5)
    pt2 = random.randint(6, 11)
    
    c1 = p1[:pt1] + p2[pt1:pt2] + p1[pt2:]
    c2 = p2[:pt1] + p1[pt1:pt2] + p2[pt2:]
    return c1, c2

def mutate(ind):
    """Mutate genes based on their specific types/ranges"""
    new_ind = ind[:]
    for i in range(12):
        if random.random() < MUTATION_RATE:
            if i < 3: # Weights (0-2)
                new_ind[i] = random.uniform(WEIGHT_MIN, WEIGHT_MAX)
            
            # MACD Blocks: Indices 3,6,9 are Fast (Int); Others are Ratios (Float)
            elif (i - 3) % 3 == 0: # Indices 3, 6, 9 (Fast Periods)
                new_ind[i] = random.randint(FAST_MIN, FAST_MAX)
            elif (i - 3) % 3 == 1: # Indices 4, 7, 10 (Slow Ratios)
                new_ind[i] = random.uniform(SLOW_RATIO_MIN, SLOW_RATIO_MAX)
            else:                  # Indices 5, 8, 11 (Signal Ratios)
                new_ind[i] = random.uniform(SIG_RATIO_MIN, SIG_RATIO_MAX)
    return new_ind

def run_genetic_algorithm(df, timeframe_label, progress_start, progress_end):
    SIGNAL_CACHE.clear()
    
    prices = df['close'].values
    returns = df['close'].pct_change().fillna(0).values
    prices_tuple = tuple(prices)
    
    if timeframe_label == '1H': factor = 365*24
    elif timeframe_label == '4H': factor = 365*6
    else: factor = 365

    # Initialize
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    
    best_overall_fitness = -999
    best_overall_ind = None
    
    for gen in range(GENERATIONS):
        pct_per_gen = (progress_end - progress_start) / GENERATIONS
        current_progress = progress_start + (gen * pct_per_gen)
        GLOBAL_STATE['progress'] = int(current_progress)
        GLOBAL_STATE['status'] = f"GA {timeframe_label}: Gen {gen+1}/{GENERATIONS} (Best Sharpe: {best_overall_fitness:.3f})"
        
        fitness_scores = []
        for ind in population:
            f = evaluate_fitness(ind, prices_tuple, returns, factor)
            fitness_scores.append((ind, f))
            if f > best_overall_fitness:
                best_overall_fitness = f
                best_overall_ind = ind[:]

        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        next_gen = [x[0] for x in fitness_scores[:ELITISM_COUNT]]
        
        while len(next_gen) < POPULATION_SIZE:
            sample = random.sample(fitness_scores, TOURNAMENT_SIZE)
            p1 = max(sample, key=lambda x: x[1])[0]
            sample = random.sample(fitness_scores, TOURNAMENT_SIZE)
            p2 = max(sample, key=lambda x: x[1])[0]
            
            c1, c2 = crossover(p1, p2)
            next_gen.append(mutate(c1))
            if len(next_gen) < POPULATION_SIZE:
                next_gen.append(mutate(c2))
        
        population = next_gen

    weights, params = decode_individual(best_overall_ind)
    s1 = get_cached_signal(prices_tuple, *params[0])
    s2 = get_cached_signal(prices_tuple, *params[1])
    s3 = get_cached_signal(prices_tuple, *params[2])
    
    total_w = sum(weights)
    pos = (weights[0]*s1 + weights[1]*s2 + weights[2]*s3) / total_w
    strat_rets = pos * returns
    
    best_params = {
        'macds': [f"{p[0]}/{p[1]}/{p[2]}" for p in params],
        'weights': [round(w, 2) for w in weights],
        'sharpe': round(best_overall_fitness, 4),
        'return': round((np.prod(1 + strat_rets) - 1) * 100, 2)
    }
    
    best_curve = pd.Series(np.cumprod(1 + strat_rets), index=df.index)
    return best_params, best_curve

# --- PLOTTING ---
def create_plot(curves_data):
    fig, axes = plt.subplots(len(curves_data), 1, figsize=(12, 5 * len(curves_data)))
    if len(curves_data) == 1: axes = [axes]
    
    for ax, (label, base_df, strat_curve) in zip(axes, curves_data):
        base_curve = (1 + base_df['close'].pct_change().fillna(0)).cumprod()
        if not base_curve.empty: base_curve = base_curve / base_curve.iloc[0]
        if not strat_curve.empty: strat_curve = strat_curve / strat_curve.iloc[0]

        ax.plot(base_curve.index, base_curve, label='BTC Buy & Hold', color='white', alpha=0.3)
        ax.plot(strat_curve.index, strat_curve, label='GA Strategy', color='#00e5ff', linewidth=1.5)
        ax.set_title(f"Equity Curve: {label}")
        ax.set_yscale('log')
        ax.set_facecolor('#1e1e1e')
        ax.grid(True, which="both", ls="-", alpha=0.1, color='white')
        ax.legend(facecolor='#1e1e1e', labelcolor='white')
        
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.title.set_color('white')
        for spine in ax.spines.values(): spine.set_color('#444')

    fig.patch.set_facecolor('#121212')
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# --- WORKER THREAD ---
def background_worker():
    try:
        GLOBAL_STATE['status'] = "Fetching Binance Data..."
        GLOBAL_STATE['progress'] = 5
        
        df_1h = fetch_binance_data(SYMBOL, '1h', START_YEAR)
        if df_1h.empty: raise Exception("No data fetched")
            
        df_4h = df_1h.resample('4h').last().dropna()
        df_1d = df_1h.resample('1D').last().dropna()
        
        res_1h, curve_1h = run_genetic_algorithm(df_1h, '1H', 10, 40)
        res_4h, curve_4h = run_genetic_algorithm(df_4h, '4H', 40, 70)
        res_1d, curve_1d = run_genetic_algorithm(df_1d, '1D', 70, 95)
        
        GLOBAL_STATE['status'] = "Generating Plots..."
        plot_b64 = create_plot([
            ('1 Hour', df_1h, curve_1h),
            ('4 Hour', df_4h, curve_4h),
            ('1 Day', df_1d, curve_1d)
        ])
        
        GLOBAL_STATE['results'] = [
            {'tf': '1H', 'macds': res_1h['macds'], 'weights': res_1h['weights'], 'sharpe': res_1h['sharpe'], 'ret': res_1h['return']},
            {'tf': '4H', 'macds': res_4h['macds'], 'weights': res_4h['weights'], 'sharpe': res_4h['sharpe'], 'ret': res_4h['return']},
            {'tf': '1D', 'macds': res_1d['macds'], 'weights': res_1d['weights'], 'sharpe': res_1d['sharpe'], 'ret': res_1d['return']}
        ]
        GLOBAL_STATE['plot'] = plot_b64
        GLOBAL_STATE['progress'] = 100
        GLOBAL_STATE['status'] = "Genetic Optimization Completed"
        GLOBAL_STATE['done'] = True
        
    except Exception as e:
        GLOBAL_STATE['error'] = str(e)
        GLOBAL_STATE['status'] = "Error Occurred"
        print(f"Background Worker Error: {e}")

# --- FLASK APP ---
@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Genetic Algo Strategy</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background: #121212; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
            .container { max-width: 1000px; margin-top: 40px; }
            .card { background: #1e1e1e; border: 1px solid #333; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
            .card-title { color: #00e5ff; }
            .progress { height: 25px; background-color: #333; border-radius: 12px; }
            .progress-bar { background-color: #00e5ff; transition: width 0.5s ease; color: #000; font-weight: bold; }
            .table { color: #e0e0e0; }
            .table-dark { --bs-table-bg: #1e1e1e; }
            th { color: #888; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2>🧬 12-Variable Genetic Optimization</h2>
                <span class="badge bg-secondary">Pop: 60 | Gens: 25</span>
            </div>
            
            <div id="progressSection" class="card p-4">
                <h5 class="card-title">Evolution Status</h5>
                <p id="statusText" class="text-muted">Initializing Genetic Algorithm...</p>
                <div class="progress mb-3">
                    <div id="progressBar" class="progress-bar progress-bar-striped progress-bar-animated" 
                         role="progressbar" style="width: 0%">0%</div>
                </div>
            </div>

            <div id="results" style="display:none;">
                <div class="card p-3">
                    <h5 class="card-title mb-3">Best Genotypes Found</h5>
                    <table class="table table-dark table-hover">
                        <thead>
                            <tr>
                                <th>Timeframe</th>
                                <th>MACD Params (Fast/Slow/Sig)</th>
                                <th>Weights</th>
                                <th>Sharpe</th>
                                <th>Return (%)</th>
                            </tr>
                        </thead>
                        <tbody id="resBody"></tbody>
                    </table>
                </div>
                <div class="card p-2 text-center">
                    <img id="perfPlot" style="max-width:100%; border-radius: 4px;">
                </div>
            </div>
        </div>

        <script>
            function pollStatus() {
                fetch('/status')
                .then(response => response.json())
                .then(data => {
                    const progressBar = document.getElementById('progressBar');
                    const statusText = document.getElementById('statusText');
                    
                    progressBar.style.width = data.progress + '%';
                    progressBar.textContent = data.progress + '%';
                    statusText.textContent = data.status;

                    if (data.error) {
                        statusText.textContent = "Error: " + data.error;
                        progressBar.classList.add('bg-danger');
                        return;
                    }

                    if (data.done) {
                        document.getElementById('progressSection').style.display = 'none';
                        const results = document.getElementById('results');
                        const tbody = document.getElementById('resBody');
                        
                        if (results.style.display === 'none') {
                            tbody.innerHTML = '';
                            data.results.forEach(r => {
                                let macds = r.macds.join('<br>');
                                let weights = r.weights.join(', ');
                                tbody.innerHTML += `<tr>
                                    <td>${r.tf}</td>
                                    <td><small>${macds}</small></td>
                                    <td><small>[${weights}]</small></td>
                                    <td class="text-info fw-bold">${r.sharpe}</td>
                                    <td>${r.ret}%</td>
                                </tr>`;
                            });
                            document.getElementById('perfPlot').src = 'data:image/png;base64,' + data.plot;
                            results.style.display = 'block';
                        }
                    } else {
                        setTimeout(pollStatus, 1000);
                    }
                })
                .catch(err => console.error(err));
            }
            window.onload = pollStatus;
        </script>
    </body>
    </html>
    """)

@app.route('/status')
def get_status():
    return jsonify(GLOBAL_STATE)

if __name__ == '__main__':
    print("Starting Background Genetic Optimization...")
    t = threading.Thread(target=background_worker)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
