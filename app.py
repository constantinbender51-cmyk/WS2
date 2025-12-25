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

from flask import Flask, send_file, jsonify, render_template_string
from gymnasium import spaces
import gymnasium as gym
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RL_Trader")

app = Flask(__name__)

# --- CONFIGURATION ---
WINDOW_SIZE = 30
TRAINING_STEPS = 20000  # Increased slightly
TRAIN_TEST_SPLIT = 0.8
INITIAL_BALANCE = 10000.0

# --- SHARED GLOBAL STATE ---
GLOBAL_STATE = {
    "status": "initializing",
    "message": "Booting up...",
    "progress": 0,
    "total_steps": TRAINING_STEPS,
    "metrics": {
        "portfolio": [],      # Full history of portfolio values
        "entropy": []         # Tracks if model is 'confused' (high) or 'sure' (low)
    },
    "error_msg": "",
    "final_plot_ready": False,
    "results": {}
}

state_lock = threading.Lock()
worker_thread_started = False

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
    logger.info("Starting batch data fetch...")
    data_frames = []
    for item in METRICS_TO_FETCH:
        series = fetch_single_metric(item['slug'])
        if series is not None:
            data_frames.append(series.to_frame(name=item['key']))
        else:
            return None

    full_df = pd.concat(data_frames, axis=1).dropna()
    
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
    feature_cols = ['price', 'hash', 'tx_count', 'revenue', 'volume', 'vol_tx_weekly']
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
        self.history = {'portfolio': [], 'benchmark': []}
        self.returns_history = [] 

    def _get_observation(self):
        obs = self.features[self.current_step - self.window_size : self.current_step]
        return obs.astype(np.float32)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action_val = int(action.item())
        else:
            action_val = int(action)

        target_leverage = LEVERAGE_MAP[action_val]
        
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, False, {}
        
        current_price_change = self.df['pct_change'].iloc[self.current_step]
        step_return = target_leverage * current_price_change
        self.portfolio_value *= (1 + step_return)
        
        # Simple Reward: Log return of portfolio
        # We use log return to treat 50% gain and 50% loss symmetrically
        reward = np.log(1 + step_return + 1e-9)

        if self.is_training:
            with state_lock:
                # Log every 50th step to keep JSON size manageable but show full history
                if self.current_step % 50 == 0:  
                    GLOBAL_STATE["metrics"]["portfolio"].append(float(self.portfolio_value))

        self.current_step += 1
        done = (self.current_step >= len(self.df) - 1)
        
        self.history['portfolio'].append(self.portfolio_value)
        if not self.history['benchmark']:
            self.history['benchmark'].append(self.initial_balance)
        else:
            prev_bench = self.history['benchmark'][-1]
            self.history['benchmark'].append(prev_bench * (1 + current_price_change))

        return self._get_observation(), reward, done, False, {}

# --- CALLBACK ---
class DashboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(DashboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Capture Entropy (Uncertainty) from the model's logs
        # SB3 calculates this internally; we can sometimes access it, 
        # but for simplicity we'll just track progress here.
        with state_lock:
            GLOBAL_STATE["progress"] = self.num_timesteps
        
        if self.num_timesteps % 500 == 0:
            logger.info(f"Step {self.num_timesteps}/{TRAINING_STEPS}")
        return True

# --- WORKER THREAD ---
def training_worker():
    logger.info("WORKER: Thread started.")
    try:
        with state_lock:
            GLOBAL_STATE["status"] = "fetching"
            GLOBAL_STATE["message"] = "Downloading blockchain data..."
        
        raw_df = fetch_and_prepare_data()
        
        if raw_df is None or raw_df.empty:
            with state_lock:
                GLOBAL_STATE["status"] = "error"
                GLOBAL_STATE["error_msg"] = "Data fetch failed"
            return

        with state_lock:
            GLOBAL_STATE["status"] = "preprocessing"
            GLOBAL_STATE["message"] = f"Preprocessing {len(raw_df)} days of data..."

        df, norm_features = preprocess_data(raw_df)
        
        split_idx = int(len(df) * TRAIN_TEST_SPLIT)
        train_df = df.iloc[:split_idx]
        train_feat = norm_features[:split_idx]
        test_df = df.iloc[split_idx:]
        test_feat = norm_features[split_idx:]

        with state_lock:
            GLOBAL_STATE["status"] = "training"
            GLOBAL_STATE["message"] = f"Training LSTM Agent ({TRAINING_STEPS} Steps)..."
        
        train_env = DummyVecEnv([lambda: CryptoTradingEnv(train_df, train_feat, is_training=True)])
        
        model = RecurrentPPO(
            "MlpLstmPolicy", 
            train_env, 
            verbose=0,
            learning_rate=3e-4,
            n_steps=256, # Increased batch size for stability
            batch_size=64,
            ent_coef=0.01,
            policy_kwargs={"lstm_hidden_size": 128, "enable_critic_lstm": True}
        )
        
        model.learn(total_timesteps=TRAINING_STEPS, callback=DashboardCallback())

        with state_lock:
            GLOBAL_STATE["status"] = "backtesting"
            GLOBAL_STATE["message"] = "Evaluating performance on unseen 2024 data..."

        test_env = CryptoTradingEnv(test_df, test_feat)
        obs, _ = test_env.reset()
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)

        # Store Final Results
        dates = test_df.index[WINDOW_SIZE : WINDOW_SIZE + len(test_env.history['portfolio'])]
        
        with state_lock:
            GLOBAL_STATE["results"] = {
                "dates": dates,
                "portfolio": test_env.history['portfolio'],
                "benchmark": test_env.history['benchmark']
            }
            GLOBAL_STATE["status"] = "completed"
            GLOBAL_STATE["message"] = "Training Finished."
            GLOBAL_STATE["final_plot_ready"] = True
            
    except Exception as e:
        logger.critical("FATAL ERROR", exc_info=True)
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
            # Return FULL history (downsampled in step function)
            "portfolio": GLOBAL_STATE["metrics"]["portfolio"], 
            "error": GLOBAL_STATE["error_msg"],
            "ready": GLOBAL_STATE["final_plot_ready"]
        })

@app.route('/result.png')
def get_result_image():
    with state_lock:
        data = GLOBAL_STATE.get("results", {})
    
    if not data:
        return "No results yet", 404

    dates = data["dates"]
    portfolio = np.array(data["portfolio"])
    benchmark = np.array(data["benchmark"])

    min_len = min(len(portfolio), len(benchmark), len(dates))
    portfolio = portfolio[:min_len]
    benchmark = benchmark[:min_len]
    dates = dates[:min_len]

    def get_sharpe(series):
        r = pd.Series(series).pct_change().dropna()
        if r.std() == 0: return 0
        return (r.mean() / r.std()) * np.sqrt(365)

    s_sharpe = get_sharpe(portfolio)
    b_sharpe = get_sharpe(benchmark)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates, portfolio, label=f'AI Strategy (Sharpe: {s_sharpe:.2f})', color='#28a745', linewidth=2)
    ax.plot(dates, benchmark, label=f'Buy & Hold (Sharpe: {b_sharpe:.2f})', color='#6c757d', linestyle='--', alpha=0.7)
    
    ax.set_title('Test Set Performance (2024-Present)', fontsize=14)
    ax.set_ylabel('Portfolio Value ($)')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100)
    img_buffer.seek(0)
    plt.close(fig)
    return send_file(img_buffer, mimetype='image/png')

# --- FRONTEND ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RL Training Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background-color: #0f111a; color: #cfd8dc; font-family: 'Segoe UI', sans-serif; padding: 20px; }
        .container { max-width: 1000px; margin: 0 auto; }
        .card { background-color: #1a1c24; border-radius: 8px; padding: 25px; margin-bottom: 20px; border: 1px solid #2f3342; }
        h1 { color: #fff; font-size: 24px; margin: 0 0 10px 0; }
        .status-pill { padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; text-transform: uppercase; float: right; }
        .pill-active { background: #3d5afe; color: white; }
        .pill-done { background: #00e676; color: #004d40; }
        .pill-error { background: #ff1744; color: white; }
        
        #progress-bg { background: #2f3342; height: 6px; border-radius: 3px; margin: 20px 0 10px 0; overflow: hidden; }
        #progress-fill { background: #3d5afe; height: 100%; width: 0%; transition: width 0.3s; }
        
        #final-img { width: 100%; border-radius: 4px; display: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <span id="badge" class="status-pill pill-active">Initializing</span>
            <h1>Training Monitor</h1>
            <div style="display:flex; justify-content:space-between; font-size:14px; color:#90a4ae;">
                <span id="msg">Preparing environment...</span>
                <span id="counts">0 / 0 Steps</span>
            </div>
            <div id="progress-bg"><div id="progress-fill"></div></div>
        </div>

        <div id="live-panel" class="card">
            <h2 style="font-size:18px; margin-top:0;">Training Episodes (Portfolio Value)</h2>
            <p style="font-size:12px; color:#607d8b;">
                *Each "loop" you see is one simulation of the dataset (2018-2023). 
                The drop means the episode ended and reset to $10k.
            </p>
            <canvas id="liveChart" height="200"></canvas>
        </div>

        <div id="result-panel" class="card" style="display:none;">
            <h2 style="font-size:18px; margin-top:0;">Final Result (Unseen Data)</h2>
            <img id="final-img" src="">
        </div>
    </div>

    <script>
        const ctx = document.getElementById('liveChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Portfolio Value',
                    data: [],
                    borderColor: '#00e676',
                    borderWidth: 1,
                    pointRadius: 0,
                    tension: 0
                }]
            },
            options: {
                responsive: true,
                animation: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: { grid: { color: '#2f3342' } }
                }
            }
        });

        function poll() {
            fetch('/api/status').then(r => r.json()).then(d => {
                // Update Header
                document.getElementById('badge').innerText = d.status;
                document.getElementById('msg').innerText = d.message;
                document.getElementById('counts').innerText = `${d.progress} / ${d.total}`;
                document.getElementById('progress-fill').style.width = (d.progress/d.total*100) + '%';
                
                // Update Live Chart
                if(d.status === 'training' && d.portfolio.length > 0) {
                    chart.data.labels = new Array(d.portfolio.length).fill('');
                    chart.data.datasets[0].data = d.portfolio;
                    chart.update();
                }

                // Show Final
                if(d.ready) {
                    document.getElementById('live-panel').style.display = 'none';
                    const res = document.getElementById('result-panel');
                    const img = document.getElementById('final-img');
                    if(res.style.display === 'none') {
                        res.style.display = 'block';
                        img.style.display = 'block';
                        document.getElementById('badge').className = 'status-pill pill-done';
                        img.src = '/result.png?t=' + Date.now();
                    }
                    return;
                }
                setTimeout(poll, 1000);
            });
        }
        poll();
    </script>
</body>
</html>
"""

if not worker_thread_started:
    t = threading.Thread(target=training_worker, daemon=True)
    t.start()
    worker_thread_started = True

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)