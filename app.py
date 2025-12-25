import io
import os
import time
import threading
import requests
import numpy as np
import pandas as pd
import matplotlib
# Set backend to Agg for server (headless) rendering
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import logging
import sys
import traceback

from flask import Flask, jsonify, render_template_string, send_file
from gymnasium import spaces
import gymnasium as gym
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO

app = Flask(__name__)

# --- CONFIGURATION ---
WINDOW_SIZE = 30
TRAINING_STEPS = 30000 
TRAIN_TEST_SPLIT = 0.8
INITIAL_BALANCE = 10000.0
TRADING_FEE = 0.001 

# --- SHARED STATE ---
GLOBAL_STATE = {
    "status": "initializing",
    "message": "Booting up...",
    "progress": 0,
    "total_steps": TRAINING_STEPS,
    "metrics": {
        "portfolio": [],      
        "mean_rewards": [],
        "actions": [0] * 11
    },
    "logs": [],  # Store logs here for the frontend
    "error_msg": "",
    "final_plot_ready": False,
    "results": {}
}

state_lock = threading.Lock()
worker_thread_started = False

# --- CUSTOM LOGGING HANDLER ---
class ListHandler(logging.Handler):
    """
    Redirects logs to the GLOBAL_STATE list so they show up in the web UI.
    """
    def emit(self, record):
        try:
            log_entry = self.format(record)
            with state_lock:
                GLOBAL_STATE["logs"].append(log_entry)
                # Keep only last 500 logs to save memory
                if len(GLOBAL_STATE["logs"]) > 500:
                    GLOBAL_STATE["logs"].pop(0)
        except Exception:
            self.handleError(record)

# Setup Logger
logger = logging.getLogger("RL_Trader")
logger.setLevel(logging.INFO)

# 1. Console Handler (Standard Output for Railway/Docker)
c_handler = logging.StreamHandler(sys.stdout)
c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)

# 2. Web List Handler (For the Dashboard)
l_handler = ListHandler()
l_format = logging.Formatter('[%(levelname)s] %(message)s')
l_handler.setFormatter(l_format)
logger.addHandler(l_handler)

# --- CONSTANTS ---
LEVERAGE_MAP = {i: i - 5 for i in range(11)}

METRICS_TO_FETCH = [
    {"slug": "market-price", "key": "price"},
    {"slug": "hash-rate", "key": "hash"},
    {"slug": "n-transactions", "key": "tx_count"},
    {"slug": "miners-revenue", "key": "revenue"},
    {"slug": "trade-volume", "key": "volume"}
]
BASE_URL = "https://api.blockchain.info/charts/{slug}"

# --- DATA FETCHING ---
def fetch_single_metric(slug):
    url = BASE_URL.format(slug=slug)
    params = {"timespan": "all", "format": "json", "sampled": "false"}
    headers = {"User-Agent": "Mozilla/5.0 (Cloud Deployment)"}
    try:
        logger.info(f"Fetching {slug}...")
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        if 'values' in data:
            df = pd.DataFrame(data['values'])
            df['date'] = pd.to_datetime(df['x'], unit='s')
            df.set_index('date', inplace=True)
            return df[df.index >= '2018-01-01']['y']
    except Exception as e:
        logger.error(f"Error fetching {slug}: {e}")
        return None

def fetch_and_prepare_data():
    logger.info("Starting batch data fetch...")
    data_frames = []
    for item in METRICS_TO_FETCH:
        series = fetch_single_metric(item['slug'])
        if series is not None:
            data_frames.append(series.to_frame(name=item['key']))
        else:
            logger.error(f"Failed to fetch {item['slug']}")
            return None

    full_df = pd.concat(data_frames, axis=1).dropna()
    
    # Feature: Volume / Transaction Ratio
    if 'volume' in full_df.columns and 'tx_count' in full_df.columns:
        vol_tx_ratio = full_df['volume'] / full_df['tx_count']
        weekly_avg = vol_tx_ratio.resample('W').mean()
        full_df['vol_tx_weekly'] = weekly_avg.reindex(full_df.index).ffill()
    else:
        full_df['vol_tx_weekly'] = 0

    full_df.dropna(inplace=True)
    logger.info(f"Data prepared: {len(full_df)} rows.")
    return full_df

def preprocess_data(df):
    df = df.copy()
    df['pct_change'] = df['price'].pct_change().fillna(0)
    df['volatility'] = df['pct_change'].rolling(window=7).std().fillna(0)
    
    feature_cols = ['price', 'hash', 'tx_count', 'revenue', 'volume', 'vol_tx_weekly', 'volatility']
    valid_cols = [c for c in feature_cols if c in df.columns]
    
    features = df[valid_cols].values
    
    # Sanity check for NaNs
    if np.isnan(features).any():
        logger.warning("NaNs detected in features! Filling with 0.")
        features = np.nan_to_num(features)

    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    return df, normalized_features

# --- GYM ENVIRONMENT ---
class CryptoTradingEnv(gym.Env):
    def __init__(self, df, features, initial_balance=10000, is_training=False):
        super(CryptoTradingEnv, self).__init__()
        self.df = df
        self.features = features
        self.n_features = features.shape[1]
        self.window_size = WINDOW_SIZE
        self.action_space = spaces.Discrete(11) 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size, self.n_features), 
            dtype=np.float32
        )
        self.initial_balance = initial_balance
        self.is_training = is_training
        self.reset_env()

    def reset(self, seed=None):
        self.reset_env()
        return self._get_observation(), {}

    def reset_env(self):
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.portfolio_value = self.initial_balance
        self.current_leverage = 0
        self.history = {'portfolio': [], 'benchmark': []}
        self.returns_window = [] 
        
    def _get_observation(self):
        obs = self.features[self.current_step - self.window_size : self.current_step]
        # Protect against NaNs in observation which kill the LSTM
        return np.nan_to_num(obs).astype(np.float32)

    def step(self, action):
        try:
            # 1. Parse Action
            if isinstance(action, np.ndarray): action_val = int(action.item())
            else: action_val = int(action)
            
            target_leverage = LEVERAGE_MAP[action_val]
            
            # 2. Update Stats
            if self.is_training:
                 with state_lock:
                     GLOBAL_STATE["metrics"]["actions"][action_val] += 1

            # 3. Check for Data End
            if self.current_step >= len(self.df) - 1:
                return self._get_observation(), 0, True, False, {}

            # 4. Market Moves
            market_return = self.df['pct_change'].iloc[self.current_step]
            
            # Sanity Check for bad data
            if np.isnan(market_return) or np.isinf(market_return):
                market_return = 0
                logger.warning(f"Bad data at step {self.current_step}")

            # Cost Calculation
            leverage_change = abs(target_leverage - self.current_leverage)
            cost_percent = leverage_change * TRADING_FEE
            
            # PnL Calculation
            gross_return = (target_leverage * market_return)
            net_return = gross_return - cost_percent
            
            self.portfolio_value *= (1 + net_return)
            self.current_leverage = target_leverage
            
            # 5. Reward & Done Logic
            done = False
            
            # BANKRUPTCY CHECK
            if self.portfolio_value < self.initial_balance * 0.4:
                done = True
                reward = -100 # Heavy penalty
            else:
                # Reward: Log Return normalized for stability
                # Using log return handles compounding better
                reward = np.log(1 + net_return + 1e-9) * 10 
                
                # Volatility Penalty
                volatility = self.df['volatility'].iloc[self.current_step]
                if abs(target_leverage) > 2 and volatility > 0.03:
                    reward -= 1.0

            # Dashboard Updates
            if self.is_training:
                with state_lock:
                    if self.current_step % 50 == 0:
                        GLOBAL_STATE["metrics"]["portfolio"].append(float(self.portfolio_value))

            # 6. Advance
            self.current_step += 1
            if self.current_step >= len(self.df) - 1:
                done = True
            
            self.history['portfolio'].append(self.portfolio_value)
            
            # Benchmark Update
            if not self.history['benchmark']:
                self.history['benchmark'].append(self.initial_balance)
            else:
                prev = self.history['benchmark'][-1]
                self.history['benchmark'].append(prev * (1 + market_return))

            return self._get_observation(), reward, done, False, {}

        except Exception as e:
            logger.error(f"Error in env step {self.current_step}: {e}")
            # Return safe values to prevent crash, but terminate episode
            return self._get_observation(), 0, True, False, {}

# --- WORKER ---
def training_worker():
    logger.info("Background Worker: Started.")
    try:
        with state_lock:
            GLOBAL_STATE["status"] = "fetching"
            GLOBAL_STATE["message"] = "Fetching Data..."
        
        raw_df = fetch_and_prepare_data()
        if raw_df is None or raw_df.empty: 
            raise Exception("Data fetch returned empty or None")

        with state_lock:
            GLOBAL_STATE["status"] = "preprocessing"
            GLOBAL_STATE["message"] = "Processing Data..."
            
        df, norm_features = preprocess_data(raw_df)
        
        split = int(len(df) * TRAIN_TEST_SPLIT)
        train_df = df.iloc[:split]
        train_feat = norm_features[:split]
        test_df = df.iloc[split:]
        test_feat = norm_features[split:]
        
        logger.info(f"Train Set: {len(train_df)} days | Test Set: {len(test_df)} days")

        with state_lock:
            GLOBAL_STATE["status"] = "training"
            GLOBAL_STATE["message"] = f"Training Model ({TRAINING_STEPS} Steps)..."
        
        train_env = DummyVecEnv([lambda: CryptoTradingEnv(train_df, train_feat, is_training=True)])
        
        logger.info("Initializing PPO LSTM...")
        model = RecurrentPPO(
            "MlpLstmPolicy", 
            train_env, 
            verbose=0,
            learning_rate=3e-4,
            n_steps=512, 
            batch_size=64,
            ent_coef=0.02, 
            gamma=0.99,
            policy_kwargs={"lstm_hidden_size": 128, "enable_critic_lstm": True}
        )
        
        class ProgCallback(BaseCallback):
            def _on_step(self):
                with state_lock: GLOBAL_STATE["progress"] = self.num_timesteps
                if self.num_timesteps % 1000 == 0:
                    logger.info(f"Step {self.num_timesteps}/{TRAINING_STEPS} completed.")
                return True
        
        logger.info("Starting .learn()...")
        model.learn(total_timesteps=TRAINING_STEPS, callback=ProgCallback())
        logger.info("Training finished.")

        with state_lock:
            GLOBAL_STATE["status"] = "backtesting"
            GLOBAL_STATE["message"] = "Backtesting on 2024+ data..."

        test_env = CryptoTradingEnv(test_df, test_feat)
        obs, _ = test_env.reset()
        done = False
        logger.info("Running backtest loop...")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = test_env.step(action)

        logger.info("Backtest complete. Generating results.")
        
        with state_lock:
            GLOBAL_STATE["results"] = {
                "dates": test_df.index[WINDOW_SIZE : WINDOW_SIZE + len(test_env.history['portfolio'])],
                "portfolio": test_env.history['portfolio'],
                "benchmark": test_env.history['benchmark']
            }
            GLOBAL_STATE["status"] = "completed"
            GLOBAL_STATE["message"] = "Analysis Done."
            GLOBAL_STATE["final_plot_ready"] = True
            
    except Exception as e:
        # Capture the FULL Traceback
        err_msg = traceback.format_exc()
        logger.error(f"FATAL WORKER CRASH:\n{err_msg}")
        with state_lock:
            GLOBAL_STATE["status"] = "error"
            GLOBAL_STATE["error_msg"] = str(e)
            # Log it to the UI console too
            GLOBAL_STATE["logs"].append(f"[FATAL] {str(e)}")

# --- FLASK ROUTES ---
@app.route('/')
def dashboard():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def get_status():
    with state_lock:
        return jsonify({
            "status": GLOBAL_STATE["status"],
            "message": GLOBAL_STATE["message"],
            "progress": GLOBAL_STATE["progress"],
            "total": GLOBAL_STATE["total_steps"],
            "portfolio": GLOBAL_STATE["metrics"]["portfolio"], 
            "actions": GLOBAL_STATE["metrics"]["actions"],
            "error": GLOBAL_STATE["error_msg"],
            "ready": GLOBAL_STATE["final_plot_ready"],
            "logs": GLOBAL_STATE["logs"][-50:] # Send last 50 logs to frontend
        })

@app.route('/result.png')
def get_result_image():
    with state_lock: data = GLOBAL_STATE.get("results", {})
    if not data: return "No results", 404

    dates = data["dates"]
    port = np.array(data["portfolio"])
    bench = np.array(data["benchmark"])
    
    L = min(len(port), len(bench), len(dates))
    dates, port, bench = dates[:L], port[:L], bench[:L]
    
    def get_sharpe(s):
        r = pd.Series(s).pct_change().dropna()
        if r.std() == 0: return 0
        return (r.mean() / r.std()) * np.sqrt(365)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates, port, label=f'AI (Sharpe: {get_sharpe(port):.2f})', color='#00e676')
    ax.plot(dates, bench, label=f'Hold (Sharpe: {get_sharpe(bench):.2f})', color='gray', linestyle='--')
    ax.set_title("Out-of-Sample Performance")
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>RL Training Console</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background:#0d1117; color:#c9d1d9; font-family:'Courier New', monospace; margin:20px; }
        .card { background:#161b22; padding:20px; margin-bottom:20px; border: 1px solid #30363d; border-radius:6px; }
        .bar-bg { background:#21262d; height:10px; border-radius:5px; overflow:hidden; }
        .bar-fill { background:#238636; height:100%; width:0%; transition:0.3s; }
        .grid { display:grid; grid-template-columns: 2fr 1fr; gap:20px; }
        
        /* TERMINAL STYLES */
        .terminal { 
            background: #000; 
            color: #0f0; 
            padding: 15px; 
            height: 200px; 
            overflow-y: auto; 
            font-size: 12px;
            border-radius: 4px;
            border: 1px solid #333;
        }
        .log-entry { margin-bottom: 2px; border-bottom: 1px solid #111; }
        .log-entry:last-child { border: none; }
        
        @media(max-width:800px) { .grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <h1 style="margin:0; font-size:20px;">RL Trading Bot Control</h1>
            <span id="status" style="font-weight:bold; color:#238636;">LOADING</span>
        </div>
        <div style="margin:15px 0;">
            <div class="bar-bg"><div id="bar" class="bar-fill"></div></div>
            <div style="display:flex; justify-content:space-between; font-size:12px; color:#8b949e; margin-top:5px;">
                <span id="msg">Initializing...</span>
                <span id="counts">0 / 0</span>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h3 style="margin-top:0;">Live System Logs</h3>
        <div id="terminal" class="terminal"></div>
    </div>

    <div class="grid" id="live-area">
        <div class="card">
            <h3>Training Portfolio</h3>
            <canvas id="portChart" height="200"></canvas>
        </div>
        <div class="card">
            <h3>Leverage Distribution</h3>
            <canvas id="actChart" height="200"></canvas>
        </div>
    </div>

    <div id="final-area" class="card" style="display:none;">
        <h3>Final Results</h3>
        <img id="final-img" style="width:100%; border-radius:4px;">
    </div>

    <script>
        const pc = new Chart(document.getElementById('portChart'), {
            type:'line', data:{datasets:[{borderColor:'#238636', borderWidth:1, radius:0, data:[]}]},
            options:{animation:false, plugins:{legend:{display:false}}, scales:{x:{display:false}, y:{grid:{color:'#21262d'}}}}
        });
        
        const ac = new Chart(document.getElementById('actChart'), {
            type:'bar', 
            data:{
                labels:['-5x','-4x','-3x','-2x','-1x','0x','1x','2x','3x','4x','5x'],
                datasets:[{backgroundColor:'#1f6feb', data:new Array(11).fill(0)}]
            },
            options:{animation:false, plugins:{legend:{display:false}}, scales:{y:{display:false}}}
        });

        const term = document.getElementById('terminal');

        setInterval(() => {
            fetch('/api/status').then(r=>r.json()).then(d => {
                document.getElementById('status').innerText = d.status.toUpperCase();
                document.getElementById('msg').innerText = d.message;
                document.getElementById('counts').innerText = `${d.progress} / ${d.total}`;
                document.getElementById('bar').style.width = (d.progress/d.total*100)+'%';
                
                // Update Logs
                if (d.logs && d.logs.length > 0) {
                    term.innerHTML = d.logs.map(l => `<div class="log-entry">${l}</div>`).join('');
                    term.scrollTop = term.scrollHeight; // Auto-scroll
                }
                
                if(d.status === 'training') {
                    pc.data.labels = new Array(d.portfolio.length).fill('');
                    pc.data.datasets[0].data = d.portfolio;
                    pc.update();
                    
                    ac.data.datasets[0].data = d.actions;
                    ac.update();
                }
                
                if(d.ready) {
                    document.getElementById('live-area').style.display = 'none';
                    document.getElementById('final-area').style.display = 'block';
                    document.getElementById('final-img').src = '/result.png?t='+Date.now();
                }
            });
        }, 1000);
    </script>
</body>
</html>
"""

if not worker_thread_started:
    t = threading.Thread(target=training_worker, daemon=True)
    t.start()
    worker_thread_started = True

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))