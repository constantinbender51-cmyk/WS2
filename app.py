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
# Increased generations and mutation to force exploration
GA_SETTINGS = {
    'POPULATION_SIZE': 40,
    'GENERATIONS': 10,
    'CROSSOVER_PROB': 0.7,
    'MUTATION_PROB': 0.4
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
            # Fetch batch
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
            if not ohlcv:
                break
            
            # Check progress
            last_ts = ohlcv[-1][0]
            if last_ts == current_ts:
                break # Avoid infinite loop if exchange returns same data
            
            current_ts = last_ts + 1
            all_ohlcv.extend(ohlcv)
            
            # Respect rate limits
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            logging.error(f"Fetch error: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # CRITICAL FIX: Ensure numeric types
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    # Drop any rows with NaN in critical columns
    df.dropna(subset=['close', 'open'], inplace=True)
    
    logging.info(f"Fetched {len(df)} valid candles.")
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
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1))
    
    # Window for III same as SMA for simplicity, or hardcoded small window (e.g., 10)
    iii_window = 14
    roll_sum_abs_diff = data['log_ret'].rolling(window=iii_window).sum().abs() 
    roll_abs_sum_diff = data['log_ret'].abs().rolling(window=iii_window).sum() 
    
    data['iii'] = roll_sum_abs_diff / (roll_abs_sum_diff + 1e-9)

    # 3. Signals (Shifted to prevent lookahead)
    prev_close = data['close'].shift(1)
    prev_open = data['open'].shift(1)
    prev_sma = data['sma'].shift(1)
    prev_iii = data['iii'].shift(1)
    
    # Direction: 1 (Long) if yesterday Green, -1 (Short) if Red
    data['direction'] = np.where(prev_close >= prev_open, 1, -1)
    
    # Entry Condition: |PrevClose - PrevSMA| > PrevSMA * a
    dist_from_sma = (prev_close - prev_sma).abs()
    threshold = prev_sma * a
    
    # Signal is active if price is OUTSIDE the band
    data['signal_active'] = dist_from_sma > threshold

    # Leverage Determination
    data['leverage'] = np.where(prev_iii > d, e, f)

    # --- Trade Simulation (Intraday) ---
    # Intraday SL Logic:
    # We assume we enter at Open.
    # Long SL Price = Open * (1 - c)
    # Short SL Price = Open * (1 + c)
    
    sl_long_price = data['open'] * (1 - c)
    sl_short_price = data['open'] * (1 + c)
    
    # Did we hit SL?
    # Long: Low < SL
    # Short: High > SL
    long_sl_hit = (data['direction'] == 1) & (data['low'] < sl_long_price)
    short_sl_hit = (data['direction'] == -1) & (data['high'] > sl_short_price)
    
    # Daily Raw Return (Open to Close)
    # Long: (Close - Open) / Open
    # Short: (Open - Close) / Open  => -1 * (Close - Open) / Open
    raw_ret = data['direction'] * (data['close'] - data['open']) / data['open']
    
    # Logic Selector
    # If !Signal -> 0
    # If Signal & SL Hit -> -c (Loss limit)
    # If Signal & No SL -> raw_ret
    
    # Construct condition masks
    no_signal = ~data['signal_active']
    sl_hit = long_sl_hit | short_sl_hit # Combined mask for simplicity since direction is mutual exclusive
    
    # Priority: No Signal (0) > SL Hit (-c) > Normal (raw)
    # Note: np.select checks strictly in order
    conditions = [
        no_signal,
        sl_hit
    ]
    
    choices = [
        0.0,
        -c
    ]
    
    data['strat_ret_no_lev'] = np.select(conditions, choices, default=raw_ret)
    
    # Apply Leverage
    data['strat_ret'] = data['strat_ret_no_lev'] * data['leverage']
    
    # Cleanup NaNs (start of data)
    data['strat_ret'].fillna(0, inplace=True)
    
    return data

def evaluate(individual, data):
    # Unpack
    a, b, c, d, e, f = individual
    
    res = calculate_metrics_vectorized(data, individual)
    
    # 1. Trade Count Check
    n_trades = (res['strat_ret'] != 0).sum()
    
    # If fewer than 10 trades in 2+ years, strategy is invalid/boring
    if n_trades < 10: 
        return -99.0, # Strong penalty
        
    # 2. Stability Metric (Sharpe-like)
    res['ym'] = res.index.to_period('M')
    monthly = res.groupby('ym')['strat_ret'].agg(['mean', 'std'])
    
    # Handle months with 0 std (flat or single trade)
    monthly_std = monthly['std'].replace(0, 1e-9)
    
    # Monthly Sharpe approximation
    sharpes = monthly['mean'] / monthly_std
    
    avg_sharpe = sharpes.mean()
    std_sharpe = sharpes.std()
    
    # If avg sharpe is negative, heavy penalty
    if avg_sharpe < 0:
        return -10.0 + avg_sharpe,
        
    # Stability = Mean / Std
    # We want high mean sharpe, low variance in sharpe
    stability = avg_sharpe / (std_sharpe + 1e-9)
    
    return stability,

# --- 3. GENETIC ALGORITHM SETUP ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def run_ga(train_df):
    toolbox = base.Toolbox()
    
    # Tweaked ranges to encourage more trading
    toolbox.register("attr_a", random.uniform, 0.001, 0.08)  # Band % (Lower min to trigger more trades)
    toolbox.register("attr_b", random.randint, 5, 50)        # SMA Period (Shorter period = more responsive)
    toolbox.register("attr_c", random.uniform, 0.01, 0.05)   # SL % (Tight SL)
    toolbox.register("attr_d", random.uniform, 0.2, 0.8)     # III Thresh
    toolbox.register("attr_e", random.uniform, 1.0, 3.0)     # High Lev
    toolbox.register("attr_f", random.uniform, 0.5, 1.0)     # Low Lev

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_a, toolbox.attr_b, toolbox.attr_c, 
                      toolbox.attr_d, toolbox.attr_e, toolbox.attr_f), n=1)
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, data=train_df)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=[0.005, 5, 0.01, 0.05, 0.2, 0.05], indpb=0.2)
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
store = {}

@app.route('/')
def report():
    if not store:
        return "Calculation pending... check terminal."
        
    # Prepare Data
    train_df = store['train']
    test_df = store['test']
    params = store['params']
    
    # Prepare Plotly Data (Explicit List Conversion)
    train_x = train_df.index.astype(str).tolist()
    train_y = train_df['cum_ret'].tolist()
    
    # Bridge the gap for visual continuity
    last_train_val = train_y[-1]
    
    test_x = test_df.index.astype(str).tolist()
    # Prepend last train point to test to connect lines
    test_x_plot = [train_x[-1]] + test_x
    test_y_plot = [last_train_val] + test_df['cum_ret'].tolist()
    
    fig = go.Figure()
    
    # Train Trace
    fig.add_trace(go.Scatter(
        x=train_x,
        y=train_y,
        mode='lines',
        name='Training',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Test Trace
    fig.add_trace(go.Scatter(
        x=test_x_plot,
        y=test_y_plot,
        mode='lines',
        name='Validation',
        line=dict(color='#2ca02c', width=2)
    ))

    fig.update_layout(
        title=f'Optimization Result: {SYMBOL} (Starting Capital: 1.0)',
        yaxis_title='Equity Multiplier',
        template='plotly_white',
        height=600,
        hovermode="x unified"
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Metrics Table
    full_df = pd.concat([train_df, test_df])
    full_df['ym'] = full_df.index.to_period('M')
    m_grouped = full_df.groupby('ym')['strat_ret'].sum()
    
    table_html = ""
    for period in sorted(m_grouped.index, reverse=True):
        val = m_grouped[period]
        bg = "#e6ffe6" if val > 0 else "#ffe6e6"
        color = "#006600" if val > 0 else "#cc0000"
        table_html += f"""
        <tr style="background-color: {bg}; color: {color};">
            <td>{period}</td>
            <td style="text-align: right; font-weight: bold;">{val*100:.2f}%</td>
        </tr>
        """

    param_html = f"""
    <ul class="params-list">
        <li><strong>Band (a):</strong> {params[0]*100:.3f}%</li>
        <li><strong>SMA (b):</strong> {int(params[1])}</li>
        <li><strong>SL (c):</strong> {params[2]*100:.2f}%</li>
        <li><strong>III Thresh (d):</strong> {params[3]:.2f}</li>
        <li><strong>High Lev (e):</strong> {params[4]:.2f}x</li>
        <li><strong>Low Lev (f):</strong> {params[5]:.2f}x</li>
    </ul>
    """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Genetic Strategy Results</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 20px; background: #f7f9fc; }}
            .container {{ max_width: 1400px; margin: 0 auto; display: grid; grid-template-columns: 3fr 1fr; gap: 20px; }}
            .card {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
            h1 {{ grid-column: 1 / -1; text-align: center; color: #2c3e50; margin-bottom: 30px; }}
            .params-list {{ list-style: none; padding: 0; }}
            .params-list li {{ padding: 10px 0; border-bottom: 1px solid #edf2f7; font-size: 0.95em; }}
            table {{ width: 100%; border-collapse: collapse; }}
            td, th {{ padding: 12px; border-bottom: 1px solid #edf2f7; }}
            .scroll-table {{ height: 600px; overflow-y: auto; }}
        </style>
    </head>
    <body>
        <h1>Genetic Algorithm Results: {SYMBOL}</h1>
        <div class="container">
            <div class="card">
                <div id="chart"></div>
                <div style="margin-top:20px; padding:15px; background:#f8f9fa; border-radius:8px;">
                    <h3>Strategy Logic</h3>
                    <p style="color:#555; line-height:1.6;">
                        1. <strong>Entry:</strong> If price deviates more than <strong>{params[0]*100:.2f}%</strong> from the <strong>{int(params[1])}-day SMA</strong>.<br>
                        2. <strong>Direction:</strong> Follow yesterday's move (Momentum).<br>
                        3. <strong>Risk:</strong> Hard Intraday Stop Loss at <strong>{params[2]*100:.2f}%</strong> from Open.<br>
                        4. <strong>Sizing:</strong> Leverage <strong>{params[4]:.1f}x</strong> if volatility quality (III) > {params[3]:.2f}, else <strong>{params[5]:.1f}x</strong>.
                    </p>
                </div>
            </div>
            
            <div class="card">
                <h3>Parameters</h3>
                {param_html}
                
                <h3 style="margin-top:30px">Monthly PnL</h3>
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
        print("Error: No data fetched.")
        exit()
        
    # B. Split
    mid = int(len(df) * 0.5)
    train_data = df.iloc[:mid].copy()
    test_data = df.iloc[mid:].copy()
    
    # C. Optimize
    best_ind = run_ga(train_data)
    logging.info(f"Optimization Complete. Best Params: {best_ind}")
    
    # D. Final Calculation & Stats
    train_res = calculate_metrics_vectorized(train_data, best_ind)
    # Start equity at 1.0
    train_res['cum_ret'] = (1 + train_res['strat_ret']).cumprod()
    
    train_trades = (train_res['strat_ret'] != 0).sum()
    final_train_equity = train_res['cum_ret'].iloc[-1]
    print(f"\n--- TRAIN RESULTS ---")
    print(f"Trades: {train_trades}")
    print(f"Final Equity: {final_train_equity:.2f}x")
    
    test_res = calculate_metrics_vectorized(test_data, best_ind)
    test_trades = (test_res['strat_ret'] != 0).sum()
    
    # Link test equity to end of train equity
    test_res['cum_ret'] = (1 + test_res['strat_ret']).cumprod() * final_train_equity
    final_test_equity = test_res['cum_ret'].iloc[-1]
    
    print(f"\n--- TEST RESULTS ---")
    print(f"Trades: {test_trades}")
    print(f"Final Equity: {final_test_equity:.2f}x")

    # E. Store & Serve
    store['train'] = train_res
    store['test'] = test_res
    store['params'] = best_ind
    
    print(f"\nSTARTING SERVER: http://localhost:{PORT}")
    app.run(host='0.0.0.0', port=PORT)
