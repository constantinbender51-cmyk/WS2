import io
import os
import json
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

from flask import Flask, send_file, jsonify, render_template_string
from gymnasium import spaces
import gymnasium as gym
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO

app = Flask(__name__)

# --- CONFIGURATION ---
WINDOW_SIZE = 30
TRAINING_STEPS = 15000  
TRAIN_TEST_SPLIT = 0.8
INITIAL_BALANCE = 10000.0

# --- SHARED GLOBAL STATE ---
# This dictionary acts as the "database" for the running process
GLOBAL_STATE = {
    "status": "initializing",  # Default state
    "message": "Booting up...",
    "progress": 0,
    "total_steps": TRAINING_STEPS,
    "metrics": {
        "rewards": [],
        "portfolio": []
    },
    "error_msg": "",
    "final_plot_ready": False,
    "results": {}
}

# Threading primitives
state_lock = threading.Lock()
worker_thread_started = False

# --- CONSTANTS & MAPS ---
LEVERAGE_MAP = {i: i - 5 for i in range(11)}

METRICS_TO_FETCH = [
    {"slug": "market-price", "key": "price"},
    {"slug": "hash-rate", "key": "hash"},
    {"slug": "n-transactions", "key": "tx_count"},
    {"slug": "miners-revenue", "key": "revenue"},
    {"slug": "trade-volume", "key": "volume"}
]
BASE_URL = "https://api.blockchain.info/charts/{slug}"

# --- DATA FETCHING UTILITIES ---
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
        print(f"Error fetching {slug}: {e}")
        return None

def fetch_and_prepare_data():
    print("Fetching data from Blockchain.com...")
    data_frames = []
    for item in METRICS_TO_FETCH:
        series = fetch_single_metric(item['slug'])
        if series is not None:
            data_frames.append(series.to_frame(name=item['key']))
        else:
            return None

    full_df = pd.concat(data_frames, axis=1).dropna()
    
    # Derived Metric: Volume/Tx
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
        # Handle numpy/tensor actions
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
        
        self.returns_history.append(step_return)
        if len(self.returns_history) > 20:
            mean_ret = np.mean(self.returns_history[-30:])
            std_ret = np.std(self.returns_history[-30:]) + 1e-9
            reward = mean_ret / std_ret
        else:
            reward = step_return

        # Live updates for dashboard
        if self.is_training:
            with state_lock:
                if self.current_step % 25 == 0:  # Update less frequently for performance
                    GLOBAL_STATE["metrics"]["portfolio"].append(float(self.portfolio_value))
                    GLOBAL_STATE["metrics"]["rewards"].append(float(reward))

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
        with state_lock:
            GLOBAL_STATE["progress"] = self.num_timesteps
        return True

# --- WORKER THREAD ---
def training_worker():
    """
    The main logic. Runs ONCE in the background.
    """
    try:
        with state_lock:
            GLOBAL_STATE["status"] = "fetching"
            GLOBAL_STATE["message"] = "Downloading blockchain data..."
        
        raw_df = fetch_and_prepare_data()
        
        if raw_df is None or raw_df.empty:
            with state_lock:
                GLOBAL_STATE["status"] = "error"
                GLOBAL_STATE["error_msg"] = "Failed to fetch data from Blockchain.com"
            return

        with state_lock:
            GLOBAL_STATE["status"] = "preprocessing"
            GLOBAL_STATE["message"] = "Calculating features..."

        df, norm_features = preprocess_data(raw_df)
        
        split_idx = int(len(df) * TRAIN_TEST_SPLIT)
        train_df = df.iloc[:split_idx]
        train_feat = norm_features[:split_idx]
        test_df = df.iloc[split_idx:]
        test_feat = norm_features[split_idx:]

        with state_lock:
            GLOBAL_STATE["status"] = "training"
            GLOBAL_STATE["message"] = f"Training LSTM PPO ({TRAINING_STEPS} steps)..."
        
        train_env = DummyVecEnv([lambda: CryptoTradingEnv(train_df, train_feat, is_training=True)])
        
        model = RecurrentPPO(
            "MlpLstmPolicy", 
            train_env, 
            verbose=0,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=64,
            ent_coef=0.01,
            policy_kwargs={"lstm_hidden_size": 64, "enable_critic_lstm": True}
        )
        
        callback = DashboardCallback()
        model.learn(total_timesteps=TRAINING_STEPS, callback=callback)

        with state_lock:
            GLOBAL_STATE["status"] = "backtesting"
            GLOBAL_STATE["message"] = "Running backtest on unseen data..."

        test_env = CryptoTradingEnv(test_df, test_feat)
        obs, _ = test_env.reset()
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)

        # Store Final Results in Memory
        dates = test_df.index[WINDOW_SIZE : WINDOW_SIZE + len(test_env.history['portfolio'])]
        
        with state_lock:
            GLOBAL_STATE["results"] = {
                "dates": dates,
                "portfolio": test_env.history['portfolio'],
                "benchmark": test_env.history['benchmark']
            }
            GLOBAL_STATE["status"] = "completed"
            GLOBAL_STATE["message"] = "Simulation Complete."
            GLOBAL_STATE["final_plot_ready"] = True
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        with state_lock:
            GLOBAL_STATE["status"] = "error"
            GLOBAL_STATE["error_msg"] = str(e)

# --- FLASK ROUTES ---

@app.route('/')
def dashboard():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def get_status():
    # Return light JSON for polling
    with state_lock:
        return jsonify({
            "status": GLOBAL_STATE["status"],
            "message": GLOBAL_STATE["message"],
            "progress": GLOBAL_STATE["progress"],
            "total": GLOBAL_STATE["total_steps"],
            "portfolio": GLOBAL_STATE["metrics"]["portfolio"][-100:], # Only send recent history
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
    
    ax.set_title('Final Strategy Performance (Out-of-Sample)', fontsize=14)
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

# --- FRONTEND TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RL Trading Bot Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; }
        .container { max-width: 1000px; margin: 0 auto; }
        .card { background-color: #1e1e1e; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .status-badge { display: inline-block; padding: 5px 10px; border-radius: 4px; font-weight: bold; text-transform: uppercase; font-size: 12px; }
        .status-initializing { background-color: #6c757d; }
        .status-fetching { background-color: #17a2b8; }
        .status-preprocessing { background-color: #6f42c1; }
        .status-training { background-color: #007bff; }
        .status-backtesting { background-color: #ffc107; color: #000; }
        .status-completed { background-color: #28a745; }
        .status-error { background-color: #dc3545; }
        
        #progress-container { width: 100%; background-color: #333; height: 10px; border-radius: 5px; margin: 15px 0; overflow: hidden; }
        #progress-bar { height: 100%; background-color: #007bff; width: 0%; transition: width 0.5s; }
        
        canvas { max-height: 300px; width: 100%; }
        #final-result-img { width: 100%; border-radius: 8px; display: none; }
        .live-panel { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h1>Auto-GPT Trader</h1>
                <span id="status-badge" class="status-badge status-initializing">Loading...</span>
            </div>
            <p id="status-message" style="color: #aaa; margin-bottom: 5px;">System check...</p>
            
            <div id="progress-container">
                <div id="progress-bar"></div>
            </div>
            <p id="progress-text" style="text-align: right; font-size: 14px; color: #aaa;">0 / 0 Steps</p>
        </div>

        <!-- Live Training Charts -->
        <div id="live-panel" class="live-panel">
            <div class="card">
                <h2>Live Portfolio Equity</h2>
                <canvas id="portfolioChart"></canvas>
            </div>
        </div>

        <!-- Final Results -->
        <div id="result-panel" class="card" style="display:none;">
            <h2>Strategy Performance</h2>
            <img id="final-result-img" src="" alt="Final Result Plot">
        </div>
    </div>

    <script>
        const ctx = document.getElementById('portfolioChart').getContext('2d');
        const portfolioChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Portfolio Value',
                    data: [],
                    borderColor: '#007bff',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                animation: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: { grid: { color: '#333' } }
                }
            }
        });

        function updateDashboard() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update Status
                    const badge = document.getElementById('status-badge');
                    badge.className = 'status-badge status-' + data.status;
                    badge.innerText = data.status.toUpperCase();
                    document.getElementById('status-message').innerText = data.message;

                    // Update Progress
                    if (data.total > 0) {
                        const pct = Math.min(100, (data.progress / data.total) * 100);
                        document.getElementById('progress-bar').style.width = pct + '%';
                        document.getElementById('progress-text').innerText = `${data.progress} / ${data.total} Steps`;
                    }

                    // Handle Completion
                    if (data.status === 'completed' && data.ready) {
                        document.getElementById('live-panel').style.display = 'none';
                        const resPanel = document.getElementById('result-panel');
                        const resImg = document.getElementById('final-result-img');
                        
                        if (resPanel.style.display === 'none') {
                            resPanel.style.display = 'block';
                            resImg.style.display = 'block';
                            resImg.src = '/result.png?t=' + new Date().getTime();
                        }
                        return; // Stop polling
                    }

                    // Update Live Chart
                    if (data.status === 'training' && data.portfolio.length > 0) {
                        const newData = data.portfolio;
                        portfolioChart.data.labels = new Array(newData.length).fill('');
                        portfolioChart.data.datasets[0].data = newData;
                        portfolioChart.update();
                    }
                })
                .catch(err => console.error(err));
        }

        setInterval(updateDashboard, 1000);
    </script>
</body>
</html>
"""

# --- INITIALIZATION ---
# This block runs when the script is loaded, ensuring the thread starts 
# regardless of whether it's imported (Gunicorn) or run directly.
if not worker_thread_started:
    t = threading.Thread(target=training_worker, daemon=True)
    t.start()
    worker_thread_started = True
    print("Background worker thread started.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)