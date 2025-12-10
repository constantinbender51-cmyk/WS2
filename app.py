import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
import random
from deap import base, creator, tools, algorithms
from flask import Flask, render_template_string
import plotly.graph_objects as go
import plotly.utils
import json
import logging

# --- CONFIGURATION ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
START_DATE_STR = '2018-01-01 00:00:00'
PORT = 8080

# GA Settings (Adjust for deeper optimization)
GA_SETTINGS = {
    'POPULATION_SIZE': 24,
    'GENERATIONS': 5,
    'CROSSOVER_PROB': 0.5,
    'MUTATION_PROB': 0.2
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. DATA FETCHING ---
def fetch_binance_data(symbol, timeframe, start_str):
    exchange = ccxt.binance()
    start_ts = exchange.parse8601(start_str)
    
    logging.info(f"Fetching {symbol} data starting from {start_str}...")
    
    ohlcv_list = []
    current_ts = start_ts
    now_ts = exchange.milliseconds()
    
    while current_ts < now_ts:
        try:
            # Fetch limit=1000 candles
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
            if not ohlcv:
                break
            
            start_ts_batch = ohlcv[0][0]
            end_ts_batch = ohlcv[-1][0]
            
            # If we didn't advance, break to prevent infinite loop
            if current_ts == end_ts_batch:
                break
                
            current_ts = end_ts_batch + 1 # Move to next ms
            ohlcv_list.extend(ohlcv)
            
            # Simple rate limit handling
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df = df[~df.index.duplicated(keep='first')] # Remove duplicates if any
    logging.info(f"Fetched {len(df)} rows of data.")
    return df

# --- 2. TRADING STRATEGY CORE ---
def calculate_metrics_vectorized(df, a, b, c, d, e, f):
    """
    a: Band % (0.01 - 0.20)
    b: SMA Period (10 - 200)
    c: Stop Loss % (0.01 - 0.10)
    d: III Threshold (0.1 - 0.9)
    e: High Leverage (1.0 - 5.0)
    f: Low Leverage (0.1 - 1.0)
    """
    data = df.copy()
    
    # 1. Indicators
    # SMA
    data['sma'] = data['close'].rolling(window=int(b)).mean()
    
    # Return (log returns for III calculation)
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1))
    
    # III (Efficiency Ratio) - using period b for simplicity
    roll_abs_sum = data['log_ret'].abs().rolling(window=int(b)).sum()
    roll_sum_abs_diff = data['log_ret'].rolling(window=int(b)).sum().abs() # Numerator usually displacement
    
    # Correction: The prompt says "abs sum log returns / sum abs log returns"
    # Numerator: abs(sum(log_ret)) -> Absolute displacement
    # Denominator: sum(abs(log_ret)) -> Sum of volatility
    data['iii'] = data['log_ret'].rolling(window=int(b)).sum().abs() / \
                  (data['log_ret'].abs().rolling(window=int(b)).sum() + 1e-9)

    # Yesterday's Return (Direction)
    data['prev_ret'] = data['close'].shift(1) - data['open'].shift(1)
    data['direction'] = np.where(data['prev_ret'] > 0, 1, -1)
    
    # 2. Entry Condition: Price leaves a% band of SMA b
    # Assuming standard envelope logic: |Close - SMA| > SMA * a
    data['dist_sma'] = abs(data['close'].shift(1) - data['sma'].shift(1))
    data['sma_threshold'] = data['sma'].shift(1) * a
    data['signal_active'] = data['dist_sma'] > data['sma_threshold']

    # 3. Leverage logic
    data['leverage'] = np.where(data['iii'].shift(1) > d, e, f)
    
    # 4. Trade Execution (Vectorized simulation)
    # We trade Open to Close
    # Direction is determined by yesterday. 
    # Long if prev_ret > 0, Short if prev_ret < 0
    
    # Base Returns without SL
    # Long: (Close - Open) / Open
    # Short: (Open - Close) / Open
    # We can simplify: Direction * (Close - Open) / Open
    
    data['daily_raw_ret'] = (data['close'] - data['open']) / data['open']
    data['strategy_raw_ret'] = data['direction'] * data['daily_raw_ret']
    
    # 5. Stop Loss Logic (Intraday)
    # Long SL: Low < Open * (1 - c)
    # Short SL: High > Open * (1 + c)
    
    # Calculate SL Prices
    data['long_sl_price'] = data['open'] * (1 - c)
    data['short_sl_price'] = data['open'] * (1 + c)
    
    # Check hits
    # If Long and Low < SL
    data['long_sl_hit'] = (data['direction'] == 1) & (data['low'] < data['long_sl_price'])
    # If Short and High > SL
    data['short_sl_hit'] = (data['direction'] == -1) & (data['high'] > data['short_sl_price'])
    
    # Calculate SL Returns
    # Long SL Ret: (SL_Price - Open) / Open = -c
    # Short SL Ret: (Open - SL_Price) / Open = -c
    # It is always -c loss if hit
    
    # Apply SL
    # If signal is not active, return is 0 (no trade)
    # If SL hit, return is -c * leverage
    # Else return is strategy_raw_ret * leverage
    
    # Conditions
    conditions = [
        ~data['signal_active'],           # No Trade
        data['long_sl_hit'],              # Long SL Hit
        data['short_sl_hit']              # Short SL Hit
    ]
    
    choices = [
        0.0,
        -c,
        -c
    ]
    
    # Base return with SL applied (unleveraged)
    data['final_daily_ret_no_lev'] = np.select(conditions, choices, default=data['strategy_raw_ret'])
    
    # Apply Leverage
    data['final_daily_ret'] = data['final_daily_ret_no_lev'] * data['leverage']
    
    # Fill NaN (start of data)
    data['final_daily_ret'].fillna(0, inplace=True)
    
    return data

def evaluate_strategy(individual, data):
    a, b, c, d, e, f = individual
    
    res = calculate_metrics_vectorized(data, a, b, c, d, e, f)
    
    # Criterion: Stable Monthly Sharpe
    res['year_month'] = res.index.to_period('M')
    
    monthly_stats = res.groupby('year_month')['final_daily_ret'].agg(['mean', 'std', 'count'])
    
    # Annualized Monthly Sharpe: (Mean / Std) * sqrt(count)
    # Avoid div by zero
    monthly_sharpes = (monthly_stats['mean'] / (monthly_stats['std'] + 1e-9)) * np.sqrt(monthly_stats['count'])
    
    # Stability Score: Mean of Monthly Sharpes / Std of Monthly Sharpes
    # We want high average sharpe and low variance between months
    
    avg_sharpe = monthly_sharpes.mean()
    std_sharpe = monthly_sharpes.std()
    
    if np.isnan(avg_sharpe): return -999,
    
    # If std is 0 (unlikely or 1 month), handle gracefully
    stability_metric = avg_sharpe / (std_sharpe + 1e-9)
    
    # Penalties for losing money overall or negative sharpe
    if avg_sharpe < 0:
        return -100 + avg_sharpe, # Severe penalty
        
    return stability_metric,

# --- 3. GENETIC ALGORITHM ---
def run_genetic_optimization(train_data):
    logging.info("Starting Genetic Algorithm Optimization...")
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Parameter Ranges
    # a: Band % [0.005, 0.10]
    # b: SMA [10, 200]
    # c: SL % [0.01, 0.15]
    # d: III Threshold [0.1, 0.9]
    # e: Lev High [1.0, 4.0]
    # f: Lev Low [0.1, 1.0]

    toolbox.register("attr_a", random.uniform, 0.005, 0.10)
    toolbox.register("attr_b", random.randint, 10, 200)
    toolbox.register("attr_c", random.uniform, 0.01, 0.15)
    toolbox.register("attr_d", random.uniform, 0.1, 0.9)
    toolbox.register("attr_e", random.uniform, 1.0, 4.0)
    toolbox.register("attr_f", random.uniform, 0.1, 1.0)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_a, toolbox.attr_b, toolbox.attr_c, 
                      toolbox.attr_d, toolbox.attr_e, toolbox.attr_f), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_strategy, data=train_data)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=[0.01, 5, 0.01, 0.1, 0.5, 0.1], indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=GA_SETTINGS['POPULATION_SIZE'])
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=GA_SETTINGS['CROSSOVER_PROB'], 
                        mutpb=GA_SETTINGS['MUTATION_PROB'], 
                        ngen=GA_SETTINGS['GENERATIONS'], 
                        stats=stats, halloffame=hof, verbose=True)

    best_ind = hof[0]
    logging.info(f"Best Parameters Found: {best_ind}")
    return best_ind

# --- 4. FLASK SERVER & PLOTTING ---
app = Flask(__name__)
results_store = {}

@app.route('/')
def dashboard():
    if not results_store:
        return "Still optimizing... check console."
    
    train_metrics = results_store['train_metrics']
    test_metrics = results_store['test_metrics']
    params = results_store['params']
    
    # 1. Equity Curve Plot - Ensure data is valid and not constant
    fig_equity = go.Figure()
    
    # Train Line - Check if cum_ret has variation
    train_cum_ret = train_metrics['cum_ret']
    if train_cum_ret.nunique() <= 1:
        logging.warning("Train cumulative returns are constant, plotting raw daily returns instead.")
        train_cum_ret = train_metrics['final_daily_ret']
    
    # Test Line - Check if cum_ret has variation
    test_cum_ret = test_metrics['cum_ret']
    if test_cum_ret.nunique() <= 1:
        logging.warning("Test cumulative returns are constant, plotting raw daily returns instead.")
        test_cum_ret = test_metrics['final_daily_ret']
    
    fig_equity.add_trace(go.Scatter(
        x=train_metrics.index, 
        y=train_cum_ret,
        mode='lines',
        name='Training Data (50%)',
        line=dict(color='blue')
    ))
    
    fig_equity.add_trace(go.Scatter(
        x=test_metrics.index, 
        y=test_cum_ret,
        mode='lines',
        name='Test Data (50%)',
        line=dict(color='green')
    ))

    fig_equity.update_layout(title='Strategy Equity Curve (Train vs Test)',
                             xaxis_title='Date', yaxis_title='Cumulative Return Multiplier')
    
    json_equity = json.dumps(fig_equity, cls=plotly.utils.PlotlyJSONEncoder)

    # 2. Daily Returns Plot (Actual Results)
    fig_daily = go.Figure()
    
    # Train Daily Returns
    fig_daily.add_trace(go.Scatter(
        x=train_metrics.index, 
        y=train_metrics['final_daily_ret'],
        mode='lines',
        name='Training Daily Returns',
        line=dict(color='blue')
    ))
    
    # Test Daily Returns
    fig_daily.add_trace(go.Scatter(
        x=test_metrics.index, 
        y=test_metrics['final_daily_ret'],
        mode='lines',
        name='Test Daily Returns',
        line=dict(color='green')
    ))

    fig_daily.update_layout(title='Strategy Daily Returns (Train vs Test)',
                            xaxis_title='Date', yaxis_title='Daily Return')
    
    json_daily = json.dumps(fig_daily, cls=plotly.utils.PlotlyJSONEncoder)

    # 2. Monthly Stats Table
    # Combine for table view
    full_metrics = pd.concat([train_metrics, test_metrics])
    full_metrics['year_month'] = full_metrics.index.to_period('M').astype(str)
    
    monthly_table = full_metrics.groupby('year_month')['final_daily_ret'].sum().reset_index()
    monthly_table.columns = ['Month', 'Monthly Return']
    monthly_table['Monthly Return'] = monthly_table['Monthly Return'].apply(lambda x: f"{x*100:.2f}%")
    
    # Generate HTML Table rows
    table_rows = ""
    for _, row in monthly_table.iterrows():
        color = "red" if "-" in row['Monthly Return'] else "green"
        table_rows += f"<tr><td>{row['Month']}</td><td style='color:{color}'>{row['Monthly Return']}</td></tr>"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Genetic Algo Trading Results</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background: #f4f4f4; }}
            .container {{ max_width: 1200px; margin: 0 auto; background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1, h2 {{ color: #333; }}
            .params-box {{ background: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .grid {{ display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Genetic Optimization Results: {SYMBOL}</h1>
            
            <div class="params-box">
                <h2>Optimized Parameters</h2>
                <ul>
                    <li><strong>a (Band %):</strong> {params[0]:.4f}</li>
                    <li><strong>b (SMA Period):</strong> {int(params[1])}</li>
                    <li><strong>c (Stop Loss %):</strong> {params[2]:.4f}</li>
                    <li><strong>d (III Threshold):</strong> {params[3]:.4f}</li>
                    <li><strong>e (High Lev):</strong> {params[4]:.2f}x</li>
                    <li><strong>f (Low Lev):</strong> {params[5]:.2f}x</li>
                </ul>
            </div>

            <div class="grid">
                <div id="chart_equity"></div>
                <div id="chart_daily"></div>
            </div>
            
            <h2>Monthly Returns</h2>
            <div style="height: 400px; overflow-y: scroll;">
                <table>
                    <thead>
                        <tr><th>Month</th><th>Return</th></tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
        </div>
        
        <script>
            var graphs_equity = {json_equity};
            Plotly.newPlot('chart_equity', graphs_equity.data, graphs_equity.layout);
            
            var graphs_daily = {json_daily};
            Plotly.newPlot('chart_daily', graphs_daily.data, graphs_daily.layout);
        </script>
    </body>
    </html>
    """
    return html

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Get Data
    df = fetch_binance_data(SYMBOL, TIMEFRAME, START_DATE_STR)
    
    if len(df) < 200:
        logging.error("Not enough data fetched. Exiting.")
        exit()

    # 2. Split Data 50/50
    split_idx = int(len(df) * 0.5)
    train_data = df.iloc[:split_idx].copy()
    test_data = df.iloc[split_idx:].copy()
    
    logging.info(f"Training Data: {len(train_data)} records")
    logging.info(f"Testing Data: {len(test_data)} records")

    # 3. Optimize
    best_ind = run_genetic_optimization(train_data)
    
    # 4. Process Results for Display
    a, b, c, d, e, f = best_ind
    
    # Recalculate full series for storage
    train_res = calculate_metrics_vectorized(train_data, a, b, c, d, e, f)
    # Ensure final_daily_ret is not all zeros or constant
    if train_res['final_daily_ret'].nunique() <= 1:
        logging.warning("Train daily returns are constant or nearly constant, check strategy logic.")
    train_res['cum_ret'] = (1 + train_res['final_daily_ret']).cumprod()
    
    # For test data, we need to be careful with cumprod chaining or just start from 1
    test_res = calculate_metrics_vectorized(test_data, a, b, c, d, e, f)
    if test_res['final_daily_ret'].nunique() <= 1:
        logging.warning("Test daily returns are constant or nearly constant, check strategy logic.")
    # Start test cum_ret from where train left off for visual continuity
    last_train_val = train_res['cum_ret'].iloc[-1]
    test_res['cum_ret'] = (1 + test_res['final_daily_ret']).cumprod() * last_train_val
    
    results_store['train_metrics'] = train_res
    results_store['test_metrics'] = test_res
    results_store['params'] = best_ind
    
    # 5. Start Server
    print(f"\nOptimization Complete. Starting Web Server at http://localhost:{PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)


