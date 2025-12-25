import io
import os
import requests
import numpy as np
import pandas as pd
import matplotlib
# Set backend to Agg for server (headless) rendering
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, send_file, make_response
from gymnasium import spaces
import gymnasium as gym
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO

app = Flask(__name__)

# --- CONFIGURATION ---
WINDOW_SIZE = 30
# Reduced steps for web demo speed (prevent timeout). 
# In production, increase this to 50,000+
TRAINING_STEPS = 8000  
TRAIN_TEST_SPLIT = 0.8
INITIAL_BALANCE = 10000.0

# Map discrete actions (0-10) to Leverage (-5 to 5)
LEVERAGE_MAP = {i: i - 5 for i in range(11)}

METRICS_TO_FETCH = [
    {"slug": "market-price", "key": "price"},
    {"slug": "hash-rate", "key": "hash"},
    {"slug": "n-transactions", "key": "tx_count"},
    {"slug": "miners-revenue", "key": "revenue"},
    {"slug": "trade-volume", "key": "volume"}
]

# --- DATA FETCHING (Cached in memory to speed up re-runs) ---
_DATA_CACHE = None

def fetch_data():
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE

    print("Fetching data from Blockchain.com...")
    base_url = "https://api.blockchain.info/charts/{slug}"
    data_frames = []

    for item in METRICS_TO_FETCH:
        url = base_url.format(slug=item['slug'])
        params = {"timespan": "all", "format": "json", "sampled": "false"}
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            df = pd.DataFrame(resp.json()['values'])
            df['date'] = pd.to_datetime(df['x'], unit='s')
            df.set_index('date', inplace=True)
            df = df.rename(columns={'y': item['key']})[['key']]
            # Filter 2018+
            df = df[df.index >= '2018-01-01']
            data_frames.append(df)
        except Exception as e:
            print(f"Error fetching {item['slug']}: {e}")
            return None

    full_df = pd.concat(data_frames, axis=1).dropna()
    
    # Feature Engineering
    vol_tx_ratio = full_df['volume'] / full_df['tx_count']
    weekly_avg = vol_tx_ratio.resample('W').mean()
    full_df['vol_tx_weekly'] = weekly_avg.reindex(full_df.index).ffill()
    full_df.dropna(inplace=True)
    
    _DATA_CACHE = full_df
    return full_df

def preprocess_data(df):
    df = df.copy()
    df['pct_change'] = df['price'].pct_change().fillna(0)
    
    feature_cols = ['price', 'hash', 'tx_count', 'revenue', 'volume', 'vol_tx_weekly']
    features = df[feature_cols].values

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
        self.action_space = spaces.Discrete(11) # -5 to +5
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
        
        current_price_change = self.df['pct_change'].iloc[self.current_step]
        step_return = target_leverage * current_price_change
        self.portfolio_value *= (1 + step_return)
        
        # Reward: Differential Sharpe (Simplified)
        self.returns_history.append(step_return)
        if len(self.returns_history) > 20:
            mean_ret = np.mean(self.returns_history[-30:])
            std_ret = np.std(self.returns_history[-30:]) + 1e-9
            reward = mean_ret / std_ret
        else:
            reward = step_return

        self.current_step += 1
        done = (self.current_step >= len(self.df) - 1)
        
        self.history['portfolio'].append(self.portfolio_value)
        
        # Benchmark logic
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
        <h1>LSTM Crypto Trader</h1>
        <p>Model: Recurrent PPO (LSTM) | Action Space: -5x to 5x Leverage</p>
        <p>Data: Blockchain.com (Price, Hash, Tx, Revenue, Vol/Tx)</p>
        <hr>
        <a href="/train_and_plot" style="
            background-color: #007bff; color: white; padding: 15px 30px; 
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
    # 1. Prepare Data
    raw_df = fetch_data()
    if raw_df is None:
        return "Error fetching data from Blockchain.com"
        
    df, norm_features = preprocess_data(raw_df)
    
    # Split
    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split_idx]
    train_feat = norm_features[:split_idx]
    test_df = df.iloc[split_idx:]
    test_feat = norm_features[split_idx:]

    # 2. Train
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
    
    model.learn(total_timesteps=TRAINING_STEPS)

    # 3. Backtest
    test_env = CryptoTradingEnv(test_df, test_feat)
    obs, _ = test_env.reset()
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)

    # 4. Plotting
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

    # Figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates, portfolio, label=f'AI Strategy (Sharpe: {s_sharpe:.2f})', color='#28a745')
    ax.plot(dates, benchmark, label=f'Buy & Hold (Sharpe: {b_sharpe:.2f})', color='#6c757d', linestyle='--')
    
    ax.set_title(f'RL Strategy vs Benchmark (Test Data)\nSteps Trained: {TRAINING_STEPS}')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()

    # Output to buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100)
    img_buffer.seek(0)
    plt.close(fig)

    return send_file(img_buffer, mimetype='image/png')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
