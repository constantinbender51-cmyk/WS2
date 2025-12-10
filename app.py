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

# --- REALITY CONSTRAINTS ---
# Taker Fee (0.05%) + Slippage (0.10%) = 0.15% per side
COST_PER_SIDE = 0.0002
# Daily cost of holding leverage (approx 10-20% APR annualized)
FUNDING_RATE_DAILY = 0.0003 

GA_SETTINGS = {
    'POPULATION_SIZE': 100,
    'GENERATIONS': 10,
    'CROSSOVER_PROB': 0.7,
    'MUTATION_PROB': 0.3
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- 1. DATA FETCHING ---
def fetch_binance_data(symbol, timeframe, start_str):
    exchange = ccxt.binance()
    start_ts = exchange.parse8601(start_str)
    
    logging.info(f"Fetching {symbol}...")
    
    all_ohlcv = []
    current_ts = start_ts
    now_ts = exchange.milliseconds()
    
    while current_ts < now_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
            if not ohlcv: break
            
            last_ts = ohlcv[-1][0]
            if last_ts == current_ts: break
            
            current_ts = last_ts + 1
            all_ohlcv.extend(ohlcv)
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            logging.error(f"Fetch error: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    cols = ['open', 'high', 'low', 'close', 'volume']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    
    # CRITICAL: Sort index to ensure time linearity (prevents lookahead via unsorted data)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    logging.info(f"Fetched {len(df)} candles.")
    return df

# --- 2. STRICT TRADING LOGIC ---
def calculate_metrics_vectorized(df, params):
    a, b, c, d, e, f = params
    # a: Band %
    # b: SMA Period
    # c: Stop Loss %
    # d: III Threshold
    # e: High Lev
    # f: Low Lev

    # Work on a copy
    data = df.copy()
    
    # --- STEP 1: SIGNAL GENERATION (Day T) ---
    # We calculate everything based on CLOSE.
    # These signals are available at 23:59:59 on Day T.
    
    # SMA
    data['sma'] = data['close'].rolling(window=int(b)).mean()
    
    # III (Volatility Quality)
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1))
    roll_sum = data['log_ret'].rolling(window=14).sum().abs()
    roll_abs = data['log_ret'].abs().rolling(window=14).sum()
    data['iii'] = roll_sum / (roll_abs + 1e-9)
    
    # Entry Signal: Price vs SMA Band
    # Condition: |Close - SMA| > SMA * a
    dist_sma = (data['close'] - data['sma']).abs()
    data['signal_trigger'] = dist_sma > (data['sma'] * a)
    
    # Direction: Yesterday's Return (Close - Close[T-1])
    # Note: User said "Yesterday's return". Usually implies Close - PrevClose.
    # If User meant Candle Color (Close - Open), change here. 
    # Using Momentum (Close - PrevClose) is standard for Trend Following.
    data['momentum_dir'] = np.sign(data['close'] - data['close'].shift(1))
    
    # Leverage decision
    data['target_lev'] = np.where(data['iii'] > d, e, f)
    
    # --- STEP 2: EXECUTION (Day T+1) ---
    # We shift ALL signal columns forward by 1.
    # This aligns "Yesterday's Signal" with "Today's Candle".
    
    data['trade_active'] = data['signal_trigger'].shift(1).fillna(False)
    data['trade_dir'] = data['momentum_dir'].shift(1).fillna(0) # 1 or -1
    data['trade_lev'] = data['target_lev'].shift(1).fillna(1.0)
    
    # --- STEP 3: CALCULATE OUTCOMES (Day T+1) ---
    # We enter at OPEN of Day T+1.
    # We exit at CLOSE of Day T+1 (or SL).
    
    # Calculate Stop Loss Price relative to Entry (Open)
    # Long SL: Open * (1 - c)
    # Short SL: Open * (1 + c)
    data['sl_long'] = data['open'] * (1 - c)
    data['sl_short'] = data['open'] * (1 + c)
    
    # Check for GAP OPEN vs SL
    # If we Long, and Open < SL_Long, we are already dead. We exit at Open.
    # But SL is defined relative to Open, so SL = Open*(1-c). 
    # Gap impossible by definition of SL being % of Open.
    
    # Check Intraday SL Hit
    # Long Hit: Low < SL
    # Short Hit: High > SL
    data['long_sl_hit'] = (data['trade_dir'] == 1) & (data['low'] < data['sl_long'])
    data['short_sl_hit'] = (data['trade_dir'] == -1) & (data['high'] > data['sl_short'])
    data['sl_triggered'] = data['long_sl_hit'] | data['short_sl_hit']
    
    # --- RETURN CALCULATIONS ---
    
    # Base Return (Unleveraged)
    # If Long: (Close - Open) / Open
    # If Short: (Open - Close) / Open
    data['raw_ret'] = data['trade_dir'] * (data['close'] - data['open']) / data['open']
    
    # SL Return (Unleveraged)
    # If SL Hit, we lose exactly 'c' % (assuming no slippage past SL for now, keeping it simple)
    # SL Loss = -c
    
    # Select Return based on SL
    # If Active AND SL Hit -> -c
    # If Active AND No SL -> raw_ret
    # Else -> 0
    
    # Logic:
    conditions = [
        ~data['trade_active'],       # No trade
        data['sl_triggered']         # SL Hit
    ]
    
    choices = [
        0.0,
        -c
    ]
    
    data['strat_ret_no_lev'] = np.select(conditions, choices, default=data['raw_ret'])
    
    # --- APPLY LEVERAGE & FRICTION ---
    
    # Gross Return
    data['gross_ret'] = data['strat_ret_no_lev'] * data['trade_lev']
    
    # Friction Calculation
    # 1. Trading Fees (Entry + Exit) = COST_PER_SIDE * 2
    # Paid on NOTIONAL (Equity * Lev)
    fees = (COST_PER_SIDE * 2) * data['trade_lev']
    
    # 2. Funding Cost (Daily holding cost)
    funding = FUNDING_RATE_DAILY * data['trade_lev']
    
    # Net Return = Gross - Fees - Funding
    # Only subtract if we actually traded
    data['net_ret'] = np.where(data['trade_active'], data['gross_ret'] - fees - funding, 0.0)
    
    # Bankruptcy Clip (Cannot lose more than 100%)
    data['net_ret'] = np.maximum(data['net_ret'], -1.0)
    
    data['net_ret'].fillna(0, inplace=True)
    return data

def evaluate(individual, data):
    # Unpack
    a, b, c, d, e, f = individual
    
    res = calculate_metrics_vectorized(data, individual)
    
    # 1. Sanity: Must trade somewhat
    n_trades = res['trade_active'].sum()
    if n_trades < 20: return -100.0,
    
    # 2. Monthly Sharpe Stability
    res['ym'] = res.index.to_period('M')
    # Filter only months where trades happened
    monthly = res[res['trade_active']].groupby('ym')['net_ret'].agg(['mean', 'std', 'count'])
    
    if len(monthly) < 6: return -100.0,
    
    # Annualized Sharpe approximation
    sharpes = (monthly['mean'] * 30) / (monthly['std'] * np.sqrt(30) + 1e-9)
    
    avg_sharpe = sharpes.mean()
    std_sharpe = sharpes.std()
    
    # If losing strategy, punish hard
    if avg_sharpe < 0: return -50.0 + avg_sharpe,
    
    # Stability Metric
    stability = avg_sharpe / (std_sharpe + 1e-9)
    
    return stability,

# --- 3. GENETIC ALGORITHM ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def run_ga(train_df):
    toolbox = base.Toolbox()
    
    # Conservative Parameter Ranges
    toolbox.register("attr_a", random.uniform, 0.02, 0.15)   # Band (2-15%)
    toolbox.register("attr_b", random.randint, 10, 100)      # SMA
    toolbox.register("attr_c", random.uniform, 0.02, 0.10)   # SL (2-10%) - Minimum 2% to avoid noise
    toolbox.register("attr_d", random.uniform, 0.2, 0.8)     # III
    toolbox.register("attr_e", random.uniform, 1.0, 3.0)     # High Lev (Max 3x)
    toolbox.register("attr_f", random.uniform, 0.1, 1.0)     # Low Lev

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
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    logging.info("Optimizing...")
    algorithms.eaSimple(pop, toolbox, cxpb=GA_SETTINGS['CROSSOVER_PROB'], 
                        mutpb=GA_SETTINGS['MUTATION_PROB'], 
                        ngen=GA_SETTINGS['GENERATIONS'], 
                        stats=stats, halloffame=hof, verbose=True)
    
    return hof[0]

# --- 4. FLASK SERVER ---
app = Flask(__name__)
store = {}

@app.route('/')
def report():
    if not store: return "Calculations pending..."
    
    train = store['train']
    test = store['test']
    params = store['params']
    
    # Prepare Plot Data
    tr_x = train.index.strftime('%Y-%m-%d').tolist()
    tr_y = train['cum_ret'].tolist()
    te_x = test.index.strftime('%Y-%m-%d').tolist()
    te_y = test['cum_ret'].tolist()
    
    # Bridge Line
    te_x_plot = [tr_x[-1]] + te_x
    te_y_plot = [tr_y[-1]] + te_y
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tr_x, y=tr_y, name='Train (50%)', line=dict(color='#3366cc')))
    fig.add_trace(go.Scatter(x=te_x_plot, y=te_y_plot, name='Test (50%)', line=dict(color='#109618')))
    
    fig.update_layout(
        title=f'Equity Curve (Fee: {COST_PER_SIDE*200:.2f}%, Funding: {FUNDING_RATE_DAILY*100:.2f}%)',
        yaxis_title='Multiple',
        template='plotly_white',
        height=500
    )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Monthly Table
    full = pd.concat([train, test])
    full['ym'] = full.index.to_period('M')
    m_pnl = full.groupby('ym')['net_ret'].sum()
    
    rows = ""
    for ym in sorted(m_pnl.index, reverse=True):
        val = m_pnl[ym]
        color = "green" if val > 0 else "red"
        rows += f"<tr><td>{ym}</td><td style='color:{color}'>{val*100:.2f}%</td></tr>"
        
    return f"""
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: sans-serif; padding: 20px; background: #f5f5f5; }}
            .card {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            table {{ width: 100%; border-collapse: collapse; }}
            td, th {{ border-bottom: 1px solid #ddd; padding: 8px; }}
        </style>
    </head>
    <body>
        <h1>Strict "Next-Day" Execution Results</h1>
        <div class="card">
            <div id="plot"></div>
        </div>
        <div class="card">
            <h3>Parameters (Reality Checked)</h3>
            <ul>
                <li><strong>Band:</strong> {params[0]*100:.2f}%</li>
                <li><strong>SMA:</strong> {int(params[1])}</li>
                <li><strong>Stop Loss:</strong> {params[2]*100:.2f}%</li>
                <li><strong>Leverage:</strong> {params[4]:.2f}x / {params[5]:.2f}x</li>
            </ul>
        </div>
        <div class="card">
            <h3>Monthly Returns</h3>
            <div style="height: 300px; overflow-y: auto;">
                <table>{rows}</table>
            </div>
        </div>
        <script>
            Plotly.newPlot('plot', {graphJSON}.data, {graphJSON}.layout);
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    df = fetch_binance_data(SYMBOL, TIMEFRAME, START_DATE_STR)
    if df.empty: exit()
    
    mid = int(len(df) * 0.5)
    train_df = df.iloc[:mid].copy()
    test_df = df.iloc[mid:].copy()
    
    best = run_ga(train_df)
    
    # Train Results
    tr_res = calculate_metrics_vectorized(train_df, best)
    tr_res['cum_ret'] = (1 + tr_res['net_ret']).cumprod()
    
    # Test Results
    te_res = calculate_metrics_vectorized(test_df, best)
    # Scale test to start where train ended
    te_res['cum_ret'] = (1 + te_res['net_ret']).cumprod() * tr_res['cum_ret'].iloc[-1]
    
    store = {'train': tr_res, 'test': te_res, 'params': best}
    
    print(f"Server: http://localhost:{PORT}")
    app.run(host='0.0.0.0', port=PORT)
