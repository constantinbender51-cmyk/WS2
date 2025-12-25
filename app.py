import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO  # LSTM PPO implementation

# --- Configuration ---
WINDOW_SIZE = 30  # Number of days the LSTM looks back
TRAIN_TEST_SPLIT = 0.8  # 80% data for training, 20% for testing
INITIAL_BALANCE = 10000.0
TRADING_FEE = 0.001  # 0.1% fee per trade (optional realism)

# Map discrete actions (0-10) to Leverage (-5 to 5)
LEVERAGE_MAP = {i: i - 5 for i in range(11)} 

METRICS_TO_FETCH = [
    {"slug": "market-price", "key": "price"},
    {"slug": "hash-rate", "key": "hash"},
    {"slug": "n-transactions", "key": "tx_count"},
    {"slug": "miners-revenue", "key": "revenue"},
    {"slug": "trade-volume", "key": "volume"}
]

# --- 1. Data Fetching & Processing (Replicated & Adapted) ---
def fetch_data():
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
            # Filter 2018+ immediately
            df = df[df.index >= '2018-01-01']
            data_frames.append(df)
        except Exception as e:
            print(f"Error fetching {item['slug']}: {e}")
            return None

    # Merge all metrics on date index
    full_df = pd.concat(data_frames, axis=1).dropna()

    # --- Derived Metric: Volume / Transactions (Weekly) ---
    # We calculate the weekly mean, then reindex back to daily + forward fill
    # so the LSTM sees the "current weekly trend" value every day.
    vol_tx_ratio = full_df['volume'] / full_df['tx_count']
    weekly_avg = vol_tx_ratio.resample('W').mean()
    
    # Realign weekly data to daily index (forward fill the weekly value for the next 7 days)
    full_df['vol_tx_weekly'] = weekly_avg.reindex(full_df.index).ffill()
    
    # Drop any remaining NaNs (likely first week)
    full_df.dropna(inplace=True)
    
    print(f"Data loaded: {len(full_df)} days.")
    return full_df

def preprocess_data(df):
    """
    Normalizes data and calculates daily returns for the environment logic.
    """
    # 1. Calculate Daily Percentage Change (Asset Return) for the simulation
    df['pct_change'] = df['price'].pct_change().fillna(0)
    
    # 2. Select Features for the LSTM
    feature_cols = ['price', 'hash', 'tx_count', 'revenue', 'volume', 'vol_tx_weekly']
    features = df[feature_cols].values

    # 3. Normalize Features (Z-Score)
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    return df, normalized_features

# --- 2. Custom Gym Environment ---
class CryptoTradingEnv(gym.Env):
    def __init__(self, df, features, initial_balance=10000):
        super(CryptoTradingEnv, self).__init__()
        
        self.df = df
        self.features = features
        self.n_features = features.shape[1]
        self.window_size = WINDOW_SIZE
        
        # Action: 0 to 10 (mapped to -5x to +5x leverage)
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
        self.current_step = self.window_size # Start after the first window
        self.portfolio_value = self.initial_balance
        self.position = 0 # Current leverage
        self.history = {'portfolio': [], 'benchmark': []}
        
        # Tracking for Sharpe Calculation
        self.returns_history = [] 

    def _get_observation(self):
        # Slice the normalized features window
        obs = self.features[self.current_step - self.window_size : self.current_step]
        return obs.astype(np.float32)

    def step(self, action):
        # 1. Determine Leverage
        target_leverage = LEVERAGE_MAP[action]
        
        # 2. Get Market Data for *Next* Day (Simulation)
        # We made a decision at 'current_step', result happens at 'current_step + 1'
        if self.current_step >= len(self.df) - 1:
            done = True
            # Final Step
            return self._get_observation(), 0, done, False, {}
        
        current_price_change = self.df['pct_change'].iloc[self.current_step]
        
        # 3. Calculate Portfolio Return
        # Simple Logic: Leverage * Asset_Return
        # (Ignoring complex margin calls/liquidation for pure strategy search)
        step_return = target_leverage * current_price_change
        
        # Update Portfolio Value
        self.portfolio_value *= (1 + step_return)
        
        # 4. Reward Engineering: Online Sharpe Ratio (Differential Sharpe)
        # We want to maximize risk-adjusted return. 
        self.returns_history.append(step_return)
        
        # If we have enough history, calculate Sharpe-like reward
        if len(self.returns_history) > 10:
            mean_ret = np.mean(self.returns_history[-30:]) # Rolling 30-day
            std_ret = np.std(self.returns_history[-30:]) + 1e-9
            sharpe = mean_ret / std_ret
            reward = sharpe 
        else:
            reward = step_return # Fallback for early steps

        # 5. Advance Step
        self.current_step += 1
        done = (self.current_step >= len(self.df) - 1)
        
        # Update History
        self.history['portfolio'].append(self.portfolio_value)
        # Benchmark: Buy and Hold (1x Leverage)
        if len(self.history['benchmark']) == 0:
            self.history['benchmark'].append(self.initial_balance)
        else:
            self.history['benchmark'].append(self.history['benchmark'][-1] * (1 + current_price_change))

        return self._get_observation(), reward, done, False, {}

# --- 3. Execution Pipeline ---

def main():
    # A. Fetch & Process
    raw_df = fetch_data()
    if raw_df is None: return
    
    df, norm_features = preprocess_data(raw_df)
    
    # B. Split Train/Test
    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    
    train_df = df.iloc[:split_idx]
    train_features = norm_features[:split_idx]
    
    test_df = df.iloc[split_idx:]
    test_features = norm_features[split_idx:]
    
    print(f"Training on {len(train_df)} days | Testing on {len(test_df)} days")

    # C. Create Environments
    train_env = DummyVecEnv([lambda: CryptoTradingEnv(train_df, train_features)])
    test_env = CryptoTradingEnv(test_df, test_features)

    # D. Train LSTM Agent (RecurrentPPO)
    print("Training LSTM Agent (this may take a moment)...")
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        train_env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=128,          # Update policy every 128 steps
        batch_size=64,
        ent_coef=0.01,        # Entropy to encourage exploration
        policy_kwargs={"lstm_hidden_size": 64, "enable_critic_lstm": True}
    )
    
    # Train for defined timesteps (increase for better results)
    model.learn(total_timesteps=30000) 
    print("Training Complete.")

    # E. Backtest on Test Data (Unseen 2024-Present)
    obs, _ = test_env.reset()
    done = False
    
    print("Running Backtest Strategy...")
    while not done:
        # LSTM requires passing cell states (lstm_states)
        # However, SB3 predict handles this internally if we don't pass explicit states for single env
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)

    # F. Visualization & Results
    history = test_env.history
    portfolio = np.array(history['portfolio'])
    benchmark = np.array(history['benchmark'])
    
    # Trim to matching lengths
    min_len = min(len(portfolio), len(benchmark))
    portfolio = portfolio[:min_len]
    benchmark = benchmark[:min_len]
    dates = test_df.index[WINDOW_SIZE : WINDOW_SIZE + min_len]

    # Calculate Metrics
    def calc_sharpe(series):
        # Approx Sharpe assuming daily series
        ret = pd.Series(series).pct_change().dropna()
        return (ret.mean() / ret.std()) * np.sqrt(365)

    strat_sharpe = calc_sharpe(portfolio)
    bench_sharpe = calc_sharpe(benchmark)

    print(f"\n--- Results ---")
    print(f"Strategy Final Balance: ${portfolio[-1]:,.2f}")
    print(f"Benchmark Final Balance: ${benchmark[-1]:,.2f}")
    print(f"Strategy Sharpe Ratio: {strat_sharpe:.2f}")
    print(f"Benchmark Sharpe Ratio: {bench_sharpe:.2f}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates, portfolio, label=f'AI Strategy (Sharpe: {strat_sharpe:.2f})', color='green')
    plt.plot(dates, benchmark, label=f'Buy & Hold (Sharpe: {bench_sharpe:.2f})', color='gray', linestyle='--')
    plt.title('LSTM RL Trading Strategy vs Buy & Hold (Out-of-Sample)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save or Show
    plt.savefig('rl_strategy_results.png')
    print("Plot saved as 'rl_strategy_results.png'")
    # plt.show() # Uncomment if running locally with display

if __name__ == "__main__":
    main()
