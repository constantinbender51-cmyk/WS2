import io
import os
import requests
import numpy as np
import pandas as pd
import matplotlib
# Set backend to Agg for server (headless) rendering
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from flask import Flask, send_file
from gymnasium import spaces
import gymnasium as gym
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO

app = Flask(__name__)

# --- CONFIGURATION ---
WINDOW_SIZE = 30
# Training steps. Increase to 50000+ for better results if time permits.
TRAINING_STEPS = 10000  
TRAIN_TEST_SPLIT = 0.8
INITIAL_BALANCE = 10000.0

# Map discrete actions (0-10) to Leverage (-5 to 5)
LEVERAGE_MAP = {i: i - 5 for i in range(11)}

# Exact metrics from your file
METRICS_TO_FETCH = [
    {"slug": "market-price", "key": "price"},
    {"slug": "hash-rate", "key": "hash"},
    {"slug": "n-transactions", "key": "tx_count"},
    {"slug": "miners-revenue", "key": "revenue"},
    {"slug": "trade-volume", "key": "volume"}
]

BASE_URL = "https://api.blockchain.info/charts/{slug}"

# --- DATA FETCHING (COPIED FROM YOUR FILE) ---
_DATA_CACHE = None

def fetch_single_metric(slug):
    """
    Fetches data using the EXACT logic from binance_analysis (8).py
    """
    url = BASE_URL.format(slug=slug)
    
    # Parameters exactly as in your working file
    params = {
        "timespan": "all",
        "format": "json", 
        "sampled": "false"
    }
    
    # User-Agent is crucial to avoid 403 Forbidden errors
    headers = {"User-Agent": "Mozilla/5.0 (Cloud Deployment)"}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if 'values' in data:
            df = pd.DataFrame(data['values'])
            df['date'] = pd.to_datetime(df['x'], unit='s')
            df.set_index('date', inplace=True)
            
            # Filter 2018+
            df = df[df.index >= '2018-01-01']
            
            # Return just the Y value (metric value)
            return df['y']
    except Exception as e:
        print(f"Error fetching {slug}: {e}")
        return None

def fetch_and_prepare_data():
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE

    print("Fetching data from Blockchain.com...")
    data_frames = []

    for item in METRICS_TO_FETCH:
        series = fetch_single_metric(item['slug'])
        if series is not None:
            # Rename series to the key (e.g., 'price', 'hash')
            df = series.to_frame(name=item['key'])
            data_frames.append(df)
        else:
            print(f"Failed to load {item['slug']}")
            return None

    # Merge all metrics on date index
    full_df = pd.concat(data_frames, axis=1).dropna()
    
    # --- Feature Engineering ---
    # Derived Metric: Volume / Transactions (Weekly)
    # We calculate the weekly mean, then reindex back to daily + forward fill
    if 'volume' in full_df.columns and 'tx_count' in full_df.columns:
        vol_tx_ratio = full_df['volume'] / full_df['tx_count']
        weekly_avg = vol_tx_ratio.resample('W').mean()
        full_df['vol_tx_weekly'] = weekly_avg.reindex(full_df.index).ffill()
    else:
        # Fallback if specific columns fail
        full_df['vol_tx_weekly'] = 0

    full_df.dropna(inplace=True)
    
    _DATA_CACHE = full_df
    return full_df

def preprocess_data(df):
    df = df.copy()
    # Daily return for environment simulation
    df['pct_change'] = df['price'].pct_change().fillna(0)
    
    feature_cols = ['price', 'hash', 'tx_count', 'revenue', 'volume', 'vol_tx_weekly']
    
    # Ensure all columns exist
    valid_cols = [c for c in feature_cols if c in df.columns]
    features = df[valid_cols].values

    # Normalize inputs for LSTM
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    return df, normalized_features

# --- GYM ENVIRONMENT ---
class CryptoTradingEnv(gym.Env):
    def __init__(self, df, features, initial_balance=10000):
        super(CryptoTradingEnv, self).__init__()
        self.df = df
        self.features = features
        self.n_features = features.shape[1]
        self.window_size = WINDOW_SIZE
        
        # Action: 0 to 10 (mapped to -5 to +5 leverage)
        self.action_space = spaces.Discrete(11) 
        
        # Observation: Matrix of shape (WINDOW_SIZE, n_features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size, self.n_features), 
            dtype=np.float32
        )
        self.initial_balance = initial_balance
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
        target_leverage = LEVERAGE_MAP[action]
        
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, False, {}
        
        # Market move happens 'tomorrow'
        current_price_change = self.df['pct_change'].iloc[self.current_step]
        
        # Calculate PnL: Leverage * %Change
        step_return = target_leverage * current_price_change
        self.portfolio_value *= (1 + step_return)
        
        # Reward: Differential Sharpe (Simplified)
        self.returns_history.append(step_return)
        if len(self.returns_history) > 20:
            mean_ret = np.mean(self.returns_history[-30:])
            std_ret = np.std(self.returns_history[-30:]) + 1e-9
            # Reward is Sharpe Ratio of rolling window
            reward = mean_ret / std_ret
        else:
            reward = step_return

        self.current_step += 1
        done = (self.current_step >= len(self.df) - 1)
        
        self.history['portfolio'].append(self.portfolio_value)
        
        # Benchmark logic (Buy & Hold 1x)
        if not self.history['benchmark']:
            self.history['benchmark'].append(self.initial_balance)
        else:
            prev_bench = self.history['benchmark'][-1]
            self.history['benchmark'].append(prev_bench * (1 + current_price_change))

        return self._get_observation(), reward, done, False, {}

# --- FLASK ROUTES ---

@app.route('/')
def home():
    return """
    <div style="font-family: sans-serif; max-width: 600px; margin: 50px auto; text-align: center;">
        <h1>LSTM Crypto Trader (Revised)</h1>
        <p>Model: Recurrent PPO (LSTM) | Action Space: -5x to 5x Leverage</p>
        <p>Data Source: Blockchain.com (Direct API Fetch)</p>
        <hr>
        <a href="/train_and_plot" style="
            background-color: #28a745; color: white; padding: 15px 30px; 
            text-decoration: none; border-radius: 5px; font-weight: bold; font-size: 18px;">
            Run Analysis & Generate Plot
        </a>
        <p style="color: #666; font-size: 12px; margin-top: 20px;">
            Note: This may take 30-60 seconds to train on the server.
        </p>
    </div>
    """

@app.route('/train_and_plot')
def run_strategy():
    # 1. Fetch Data
    raw_df = fetch_and_prepare_data()
    if raw_df is None or raw_df.empty:
        return "Error: Could not fetch data. Please check logs."
        
    df, norm_features = preprocess_data(raw_df)
    
    # Split Data
    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split_idx]
    train_feat = norm_features[:split_idx]
    test_df = df.iloc[split_idx:]
    test_feat = norm_features[split_idx:]

    # 2. Train Model
    # Create Vectorized Environment for SB3
    train_env = DummyVecEnv([lambda: CryptoTradingEnv(train_df, train_feat)])
    
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
    
    # Train
    model.learn(total_timesteps=TRAINING_STEPS)

    # 3. Backtest
    test_env = CryptoTradingEnv(test_df, test_feat)
    obs, _ = test_env.reset()
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)

    # 4. Plot Results
    history = test_env.history
    portfolio = np.array(history['portfolio'])
    benchmark = np.array(history['benchmark'])
    
    # Align lengths
    min_len = min(len(portfolio), len(benchmark))
    portfolio = portfolio[:min_len]
    benchmark = benchmark[:min_len]
    dates = test_df.index[WINDOW_SIZE : WINDOW_SIZE + min_len]

    # Calculate Sharpe
    def get_sharpe(series):
        r = pd.Series(series).pct_change().dropna()
        if r.std() == 0: return 0
        return (r.mean() / r.std()) * np.sqrt(365)

    s_sharpe = get_sharpe(portfolio)
    b_sharpe = get_sharpe(benchmark)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates, portfolio, label=f'AI Strategy (Sharpe: {s_sharpe:.2f})', color='#007bff')
    ax.plot(dates, benchmark, label=f'Buy & Hold (Sharpe: {b_sharpe:.2f})', color='#6c757d', linestyle='--')
    
    ax.set_title(f'RL Strategy vs Benchmark (Test Data)\nSteps Trained: {TRAINING_STEPS}')
    ax.set_ylabel('Portfolio Value ($)')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()

    # Save to buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100)
    img_buffer.seek(0)
    plt.close(fig)

    return send_file(img_buffer, mimetype='image/png')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)