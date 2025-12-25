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

from flask import Flask, jsonify, render_template_string, send_file
from gymnasium import spaces
import gymnasium as gym
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RL_Trader")

app = Flask(__name__)

# --- CONFIGURATION ---
WINDOW_SIZE = 30
TRAINING_STEPS = 30000  # ~10 passes through the dataset
TRAIN_TEST_SPLIT = 0.8
INITIAL_BALANCE = 10000.0
TRADING_FEE = 0.001  # 0.1% fee per leverage unit changed

# --- SHARED STATE ---
GLOBAL_STATE = {
    "status": "initializing",
    "message": "Booting up...",
    "progress": 0,
    "total_steps": TRAINING_STEPS,
    "metrics": {
        "portfolio": [],      
        "mean_rewards": [],   # To track learning progress
        "actions": [0] * 11   # To track what moves it's choosing
    },
    "error_msg": "",
    "final_plot_ready": False,
    "results": {}
}

state_lock = threading.Lock()
worker_thread_started = False

# Mapping: 0->-5, 5->0, 10->+5
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
    logger.info("Fetching Blockchain.com data...")
    data_frames = []
    for item in METRICS_TO_FETCH:
        series = fetch_single_metric(item['slug'])
        if series is not None:
            data_frames.append(series.to_frame(name=item['key']))
        else:
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
    return full_df

def preprocess_data(df):
    df = df.copy()
    df['pct_change'] = df['price'].pct_change().fillna(0)
    
    # Add rolling volatility for the state (helps agent see risk)
    df['volatility'] = df['pct_change'].rolling(window=7).std().fillna(0)
    
    feature_cols = ['price', 'hash', 'tx_count', 'revenue', 'volume', 'vol_tx_weekly', 'volatility']
    valid_cols = [c for c in feature_cols if c in df.columns]
    
    features = df[valid_cols].values
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
        return obs.astype(np.float32)

    def step(self, action):
        # 1. Parse Action
        if isinstance(action, np.ndarray): action_val = int(action.item())
        else: action_val = int(action)
        
        target_leverage = LEVERAGE_MAP[action_val]
        
        # 2. Update Stats (Dashboard)
        if self.is_training:
             with state_lock:
                 GLOBAL_STATE["metrics"]["actions"][action_val] += 1

        # 3. Check for Data End
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, False, {}

        # 4. Execute Trade Logic
        # Calculate Cost to change position
        leverage_change = abs(target_leverage - self.current_leverage)
        cost_percent = leverage_change * TRADING_FEE
        
        # Market Move
        market_return = self.df['pct_change'].iloc[self.current_step]
        
        # Portfolio Move: (Leverage * Market) - Costs
        gross_return = (target_leverage * market_return)
        net_return = gross_return - cost_percent
        
        self.portfolio_value *= (1 + net_return)
        self.current_leverage = target_leverage
        
        # 5. Reward Engineering (The most important part)
        
        # A. Bankruptcy Penalty (Survival Constraint)
        # If we lose 50% of the account, kill the episode.
        if self.portfolio_value < self.initial_balance * 0.5:
            done = True
            reward = -100  # massive penalty
            return self._get_observation(), reward, done, False, {}
            
        # B. Sharpe-Optimized Reward
        # Reward = Return - (Volatility Penalty)
        # We scale return by 100 to make it significant for the Neural Net
        reward = (net_return * 100) 
        
        # Penalize holding max leverage if market is volatile (soft constraint)
        volatility = self.df['volatility'].iloc[self.current_step]
        if abs(target_leverage) > 3 and volatility > 0.02:
            reward -= 0.5 # discourage reckless leverage in high vol
            
        # Update Dashboard
        if self.is_training:
            with state_lock:
                if self.current_step % 50 == 0:
                    GLOBAL_STATE["metrics"]["portfolio"].append(float(self.portfolio_value))
                    # Also track rolling mean reward roughly
                    GLOBAL_STATE["metrics"]["mean_rewards"].append(float(reward))

        # 6. Advance
        self.current_step += 1
        done = (self.current_step >= len(self.df) - 1)
        
        self.history['portfolio'].append(self.portfolio_value)
        
        # Benchmark
        if not self.history['benchmark']:
            self.history['benchmark'].append(self.initial_balance)
        else:
            prev = self.history['benchmark'][-1]
            self.history['benchmark'].append(prev * (1 + market_return))

        return self._get_observation(), reward, done, False, {}

# --- WORKER ---
def training_worker():
    logger.info("WORKER: Started.")
    try:
        with state_lock:
            GLOBAL_STATE["status"] = "fetching"
            GLOBAL_STATE["message"] = "Fetching Data..."
        
        raw_df = fetch_and_prepare_data()
        if raw_df is None or raw_df.empty: raise Exception("No Data")

        with state_lock:
            GLOBAL_STATE["status"] = "preprocessing"
            GLOBAL_STATE["message"] = "Processing..."
            
        df, norm_features = preprocess_data(raw_df)
        
        split = int(len(df) * TRAIN_TEST_SPLIT)
        train_df = df.iloc[:split]
        train_feat = norm_features[:split]
        test_df = df.iloc[split:]
        test_feat = norm_features[split:]

        with state_lock:
            GLOBAL_STATE["status"] = "training"
            GLOBAL_STATE["message"] = f"Optimizing Sharpe ({TRAINING_STEPS} Steps)..."
        
        train_env = DummyVecEnv([lambda: CryptoTradingEnv(train_df, train_feat, is_training=True)])
        
        model = RecurrentPPO(
            "MlpLstmPolicy", 
            train_env, 
            verbose=0,
            learning_rate=3e-4,
            n_steps=512, # Long batch size for stable updates
            batch_size=64,
            ent_coef=0.02, # High entropy to encourage trying different leverages
            gamma=0.99,
            policy_kwargs={"lstm_hidden_size": 128, "enable_critic_lstm": True}
        )
        
        # Callback to update progress bar
        class ProgCallback(BaseCallback):
            def _on_step(self):
                with state_lock: GLOBAL_STATE["progress"] = self.num_timesteps
                return True
        
        model.learn(total_timesteps=TRAINING_STEPS, callback=ProgCallback())

        with state_lock:
            GLOBAL_STATE["status"] = "backtesting"
            GLOBAL_STATE["message"] = "Final Test Run..."

        test_env = CryptoTradingEnv(test_df, test_feat)
        obs, _ = test_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = test_env.step(action)

        with state_lock:
            GLOBAL_STATE["results"] = {
                "dates": test_df.index[WINDOW_SIZE : WINDOW_SIZE + len(test_env.history['portfolio'])],
                "portfolio": test_env.history['portfolio'],
                "benchmark": test_env.history['benchmark']
            }
            GLOBAL_STATE["status"] = "completed"
            GLOBAL_STATE["message"] = "Done."
            GLOBAL_STATE["final_plot_ready"] = True
            
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
        with state_lock:
            GLOBAL_STATE["status"] = "error"
            GLOBAL_STATE["error_msg"] = str(e)

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
            "rewards": GLOBAL_STATE["metrics"]["mean_rewards"],
            "actions": GLOBAL_STATE["metrics"]["actions"],
            "error": GLOBAL_STATE["error_msg"],
            "ready": GLOBAL_STATE["final_plot_ready"]
        })

@app.route('/result.png')
def get_result_image():
    with state_lock: data = GLOBAL_STATE.get("results", {})
    if not data: return "No results", 404

    dates = data["dates"]
    port = np.array(data["portfolio"])
    bench = np.array(data["benchmark"])
    
    # Slice to same length
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
    <title>RL Training</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background:#121212; color:#eee; font-family:sans-serif; margin:20px; }
        .card { background:#1e1e1e; padding:20px; margin-bottom:20px; border-radius:8px; }
        .bar-bg { background:#333; height:8px; border-radius:4px; overflow:hidden; }
        .bar-fill { background:#00e676; height:100%; width:0%; transition:0.3s; }
        h2 { margin-top:0; font-size:16px; color:#aaa; }
        .grid { display:grid; grid-template-columns: 1fr 1fr; gap:20px; }
        @media(max-width:600px) { .grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="card">
        <div style="display:flex; justify-content:space-between;">
            <h1 style="margin:0; font-size:20px;">Deep RL Trader</h1>
            <span id="status" style="font-weight:bold; color:#00e676;">Loading...</span>
        </div>
        <p id="msg" style="color:#777; font-size:14px;">...</p>
        <div class="bar-bg"><div id="bar" class="bar-fill"></div></div>
    </div>

    <div class="grid" id="live-area">
        <div class="card">
            <h2>Portfolio Value (Training Episodes)</h2>
            <canvas id="portChart"></canvas>
        </div>
        <div class="card">
            <h2>Action Distribution (Leverage Used)</h2>
            <canvas id="actChart"></canvas>
        </div>
    </div>

    <div id="final-area" class="card" style="display:none;">
        <h2>Final Performance (Test Data)</h2>
        <img id="final-img" style="width:100%; border-radius:4px;">
    </div>

    <script>
        const pc = new Chart(document.getElementById('portChart'), {
            type:'line', data:{datasets:[{borderColor:'#00e676', borderWidth:1, radius:0, data:[]}]},
            options:{animation:false, plugins:{legend:{display:false}}, scales:{x:{display:false}, y:{grid:{color:'#333'}}}}
        });
        
        const ac = new Chart(document.getElementById('actChart'), {
            type:'bar', 
            data:{
                labels:['-5x','-4x','-3x','-2x','-1x','0x','1x','2x','3x','4x','5x'],
                datasets:[{backgroundColor:'#3d5afe', data:new Array(11).fill(0)}]
            },
            options:{animation:false, plugins:{legend:{display:false}}, scales:{y:{display:false}}}
        });

        setInterval(() => {
            fetch('/api/status').then(r=>r.json()).then(d => {
                document.getElementById('status').innerText = d.status.toUpperCase();
                document.getElementById('msg').innerText = d.message;
                document.getElementById('bar').style.width = (d.progress/d.total*100)+'%';
                
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