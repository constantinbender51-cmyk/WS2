import ccxt
import pandas as pd
import numpy as np
import time
import random
import logging
import json
from datetime import datetime
from deap import base, creator, tools, algorithms
from flask import Flask
import plotly.graph_objects as go
import plotly.utils

# --- CONFIGURATION ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE_STR = '2018-01-01 00:00:00'
PORT = 8080

# Optimization Settings
GA_SETTINGS = {
    'POPULATION_SIZE': 30,
    'GENERATIONS': 8,
    'CROSSOVER_PROB': 0.6,
    'MUTATION_PROB': 0.3
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- 1. ROBUST DATA FETCHING ---
def fetch_binance_data(symbol, timeframe, start_str):
    exchange = ccxt.binance()
    start_ts = exchange.parse8601(start_str)
    
    logging.info(f"Fetching {symbol} data starting from {start_str}...")
    
    all_ohlcv = []
    current_ts = start_ts
    now_ts = exchange.milliseconds()
    
    while current_ts < now_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
            if not ohlcv:
                break
            
            current_ts = ohlcv[-1][0] + 1
            all_ohlcv.extend(ohlcv)
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            logging.error(f"Fetch error: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    logging.info(f"Fetched {len(df)} candles.")
    return df

# --- 2. CORE TRADING LOGIC ---
def calculate_metrics_vectorized(df, params):
    a, b, c, d, e, f = params
    # a: Band %
    # b: SMA Period
    # c: SL %
    # d: III Threshold
    # e: High Lev
    # f: Low Lev

    data = df.copy()
    
    # --- Indicators ---
    # 1. SMA (b)
    data['sma'] = data['close'].rolling(window=int(b)).mean()
    
    # 2. III (Intraday Intensity / Efficiency)
    # Computed on Log Returns over period b
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1))
    
    roll_sum_abs_diff = data['log_ret'].rolling(window=int(b)).sum().abs() # |Sum|
    roll_abs_sum_diff = data['log_ret'].abs().rolling(window=int(b)).sum() # Sum(|x|)
    
    # Avoid division by zero
    data['iii'] = roll_sum_abs_diff / (roll_abs_sum_diff + 1e-9)

    # 3. Signals (Shifted to prevent lookahead)
    # We trade TODAY based on YESTERDAY's data
    prev_close = data['close'].shift(1)
    prev_open = data['open'].shift(1)
    prev_sma = data['sma'].shift(1)
    prev_iii = data['iii'].shift(1)
    
    # Direction: Follow yesterday's move
    # If yesterday Green (Close > Open) -> Long (1)
    # If yesterday Red (Close < Open) -> Short (-1)
    data['direction'] = np.where(prev_close > prev_open, 1, -1)
    
    # Entry Condition: Price left a% band of SMA
    # |PrevClose - PrevSMA| > PrevSMA * a
    dist_from_sma = (prev_close - prev_sma).abs()
    threshold = prev_sma * a
    data['signal_active'] = dist_from_sma > threshold

    # Leverage Determination
    data['leverage'] = np.where(prev_iii > d, e, f)

    # --- Trade Simulation ---
    
    # Stop Loss Prices (Intraday)
    # Long SL: Open * (1 - c)
    # Short SL: Open * (1 + c)
    data['sl_long_price'] = data['open'] * (1 - c)
    data['sl_short_price'] = data['open'] * (1 + c)
    
    # Check for SL Hits using Low/High of current day
    # Long Hit: Low < SL
    # Short Hit: High > SL
    long_sl_hit = (data['direction'] == 1) & (data['low'] < data['sl_long_price'])
    short_sl_hit = (data['direction'] == -1) & (data['high'] > data['sl_short_price'])
    
    # Calculate Raw Return (Unleveraged)
    # Standard: Direction * (Close - Open) / Open
    # SL Hit: -c (The loss is exactly c%)
    
    raw_ret_daily = data['direction'] * (data['close'] - data['open']) / data['open']
    
    # Vectorized Selection of Return
    # Priority: 
    # 1. No Signal -> 0
    # 2. SL Hit -> -c
    # 3. Normal Close -> raw_ret_daily
    
    conditions = [
        ~data['signal_active'],
        long_sl_hit,
        short_sl_hit
    ]
    
    choices = [
        0.0,
        -c,
        -c
    ]
    
    data['strat_ret_no_lev'] = np.select(conditions, choices, default=raw_ret_daily)
    
    # Apply Leverage
    data['strat_ret'] = data['strat_ret_no_lev'] * data['leverage']
    
    # Fill NaN at start
    data['strat_ret'].fillna(0, inplace=True)
    
    return data

def evaluate(individual, data):
    # Unpack
    a, b, c, d, e, f = individual
    
    res = calculate_metrics_vectorized(data, individual)
    
    # --- Fitness Logic ---
    
    # 1. Activity Check
    # Count non-zero return days (trades)
    n_trades = (res['strat_ret'] != 0).sum()
    
    # HEAVY PENALTY for not trading. 
    # If the strategy doesn't trade, it produces a flat line.
    if n_trades < 20: 
        return -10.0, # Immediate failure
        
    # 2. Monthly Stability
    res['ym'] = res.index.to_period('M')
    monthly = res.groupby('ym')['strat_ret'].agg(['mean', 'std', 'count'])
    
    # Annualized Sharpe per month approx
    # (Mean daily ret * 30) / (Std daily * sqrt(30)) -> simplified to mean/std
    # We use a simple Sharpe-like ratio: Mean / Std
    
    sharpes = monthly['mean'] / (monthly['std'] + 1e-9)
    
    avg_sharpe = sharpes.mean()
    std_sharpe = sharpes.std()
    
    # 3. Stability Metric
    # We want High Avg Sharpe, Low Std of Sharpe
    metric = avg_sharpe / (std_sharpe + 1e-9)
    
    # Penalty for negative expectancy
    if avg_sharpe < 0:
        return -5.0 + avg_sharpe,

    return metric,

# --- 3. GENETIC ALGORITHM SETUP ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def run_ga(train_df):
    toolbox = base.Toolbox()
    
    # Gene Definitions
    toolbox.register("attr_a", random.uniform, 0.005, 0.15)  # Band %
    toolbox.register("attr_b", random.randint, 10, 100)      # SMA Period
    toolbox.register("attr_c", random.uniform, 0.01, 0.10)   # SL %
    toolbox.register("attr_d", random.uniform, 0.2, 0.8)     # III Thresh
    toolbox.register("attr_e", random.uniform, 1.0, 3.0)     # High Lev
    toolbox.register("attr_f", random.uniform, 0.1, 0.9)     # Low Lev

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_a, toolbox.attr_b, toolbox.attr_c, 
                      toolbox.attr_d, toolbox.attr_e, toolbox.attr_f), n=1)
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, data=train_df)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=[0.01, 5, 0.01, 0.05, 0.2, 0.05], indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=GA_SETTINGS['POPULATION_SIZE'])
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    logging.info("Starting Evolutionary Process...")
    algorithms.eaSimple(pop, toolbox, cxpb=GA_SETTINGS['CROSSOVER_PROB'], 
                        mutpb=GA_SETTINGS['MUTATION_PROB'], 
                        ngen=GA_SETTINGS['GENERATIONS'], 
                        stats=stats, halloffame=hof, verbose=True)
    
    return hof[0]

# --- 4. WEB SERVER ---
app = Flask(__name__)
# Global store for results
store = {}

@app.route('/')
def report():
    if not store:
        return "Calculation pending..."
        
    # Prepare Data
    train_df = store['train']
    test_df = store['test']
    params = store['params']
    
    # --- PLOTLY CHART ---
    # Concatenate for continuous timeline view, but color code
    
    fig = go.Figure()
    
    # Train Trace
    fig.add_trace(go.Scatter(
        x=train_df.index.astype(str),
        y=train_df['cum_ret'],
        mode='lines',
        name='Training Phase',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Test Trace
    # To make it look connected, we add the last point of train to test
    test_x = [train_df.index[-1]] + list(test_df.index)
    test_y = [train_df['cum_ret'].iloc[-1]] + list(test_df['cum_ret'])
    
    # Convert timestamps to str for JSON serialization safety
    test_x_str = [str(x) for x in test_x]
    
    fig.add_trace(go.Scatter(
        x=test_x_str,
        y=test_y,
        mode='lines',
        name='Validation Phase',
        line=dict(color='#2ca02c', width=2)
    ))

    fig.update_layout(
        title=f'Strategy Performance: {SYMBOL}',
        yaxis_title='Equity Multiplier (Start = 1.0)',
        template='plotly_white',
        height=600,
        hovermode="x unified"
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # --- METRICS TABLE ---
    full_df = pd.concat([train_df, test_df])
    full_df['ym'] = full_df.index.to_period('M')
    
    # Group by Month
    m_grouped = full_df.groupby('ym')['strat_ret'].sum()
    
    table_html = ""
    # Reverse order for newest first
    for period in sorted(m_grouped.index, reverse=True):
        val = m_grouped[period]
        color = "#ffcccc" if val < 0 else "#ccffcc"
        text_color = "#b30000" if val < 0 else "#006600"
        table_html += f"""
        <tr style="background-color: {color}; color: {text_color};">
            <td>{period}</td>
            <td style="text-align: right; font-weight: bold;">{val*100:.2f}%</td>
        </tr>
        """

    # Parameters List
    param_html = f"""
    <ul class="params-list">
        <li><strong>Band (a):</strong> {params[0]*100:.2f}%</li>
        <li><strong>SMA Period (b):</strong> {int(params[1])}</li>
        <li><strong>Stop Loss (c):</strong> {params[2]*100:.2f}%</li>
        <li><strong>III Thresh (d):</strong> {params[3]:.2f}</li>
        <li><strong>High Lev (e):</strong> {params[4]:.2f}x</li>
        <li><strong>Low Lev (f):</strong> {params[5]:.2f}x</li>
    </ul>
    """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crypto GA Strategy</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }}
            .container {{ max_width: 1200px; margin: 0 auto; display: grid; grid-template-columns: 3fr 1fr; gap: 20px; }}
            .card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            h1 {{ grid-column: 1 / -1; text-align: center; color: #333; }}
            .params-list {{ list-style: none; padding: 0; }}
            .params-list li {{ padding: 8px 0; border-bottom: 1px solid #eee; }}
            table {{ width: 100%; border-collapse: collapse; }}
            td, th {{ padding: 10px; border: 1px solid #ddd; }}
            .scroll-table {{ height: 600px; overflow-y: auto; }}
        </style>
    </head>
    <body>
        <h1>Genetic Algorithm Optimization Results</h1>
        <div class="container">
            <div class="card">
                <div id="chart"></div>
                <div style="margin-top: 20px;">
                    <h3>Logic Summary</h3>
                    <p>When price leaves <strong>{params[0]*100:.1f}%</strong> band of <strong>SMA {int(params[1])}</strong>, 
                    trade in direction of yesterday's momentum. 
                    Leverage is <strong>{params[4]:.2f}x</strong> if III > {params[3]:.2f}, else <strong>{params[5]:.2f}x</strong>.
                    Intraday Stop Loss at <strong>{params[2]*100:.1f}%</strong>.</p>
                </div>
            </div>
            
            <div class="card">
                <h3>Parameters</h3>
                {param_html}
                
                <h3>Monthly Returns</h3>
                <div class="scroll-table">
                    <table>
                        {table_html}
                    </table>
                </div>
            </div>
        </div>
        <script>
            var graphs = {graphJSON};
            Plotly.newPlot('chart', graphs.data, graphs.layout);
        </script>
    </body>
    </html>
    """

# --- 5. MAIN ---
if __name__ == "__main__":
    # A. Fetch
    df = fetch_binance_data(SYMBOL, TIMEFRAME, START_DATE_STR)
    if df.empty:
        print("No data fetched.")
        exit()
        
    # B. Split
    mid = int(len(df) * 0.5)
    train_data = df.iloc[:mid].copy()
    test_data = df.iloc[mid:].copy()
    
    # C. Optimize
    best_ind = run_ga(train_data)
    
    # D. Final Calculation
    train_res = calculate_metrics_vectorized(train_data, best_ind)
    train_res['cum_ret'] = (1 + train_res['strat_ret']).cumprod()
    
    test_res = calculate_metrics_vectorized(test_data, best_ind)
    
    # Link test cumulative return to end of train
    last_val = train_res['cum_ret'].iloc[-1]
    test_res['cum_ret'] = (1 + test_res['strat_ret']).cumprod() * last_val
    
    # E. Store & Serve
    store['train'] = train_res
    store['test'] = test_res
    store['params'] = best_ind
    
    print(f"\n Server running at http://localhost:{PORT}")
    app.run(host='0.0.0.0', port=PORT)
