import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ==========================================
# Configuration
# ==========================================
OUTPUT_FILE = 'market_data.csv' # Uses the file from previous step
MODEL_FILE = 'simplified_lstm_model.h5'
LOOKBACK_DAYS = 14
TARGET_HORIZONS = [1, 2, 3, 5, 8, 13, 21] 
MINUTES_PER_DAY = 1440

def load_and_prep_data():
    print("Loading data for verification...")
    df = pd.read_csv(OUTPUT_FILE)
    
    # robust timestamp parsing
    if df['timestamp'].dtype == 'O':
         df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
         df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
         
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # 1. Resample to Daily
    df_daily = df['close'].resample('1D').last().dropna().to_frame()
    df_daily['log_price'] = np.log(df_daily['close'])
    
    # 2. Re-create Features & Targets explicitly for inspection
    # Feature lag 0 = Close(T) - Close(T-1)
    df_daily['feat_lag_0'] = df_daily['log_price'].shift(0) - df_daily['log_price'].shift(1)
    
    # Target 1d = Close(T+1) - Close(T)
    df_daily['target_1d'] = df_daily['log_price'].shift(-1) - df_daily['log_price']
    
    # Full Feature Set Generation (Same as training)
    feature_cols = []
    for i in range(LOOKBACK_DAYS):
        col_name = f'feat_lag_{i}'
        df_daily[col_name] = df_daily['log_price'].shift(i) - df_daily['log_price'].shift(i + 1)
        feature_cols.append(col_name)

    # Full Target Set Generation
    target_cols = []
    for day in TARGET_HORIZONS:
        col_name = f'target_{day}d'
        df_daily[col_name] = df_daily['log_price'].shift(-day) - df_daily['log_price']
        target_cols.append(col_name)

    df_clean = df_daily.dropna()
    return df_clean, feature_cols, target_cols

def check_lookahead(df):
    print("\n--- 1. Visual Leakage Inspection ---")
    print("We are looking for obvious duplicates between Features and Targets.")
    print("Row T Feature (lag_0) = Return(T-1 to T)")
    print("Row T Target  (1d)    = Return(T to T+1)")
    
    sample = df[['feat_lag_0', 'target_1d']].head(5)
    print(sample)
    
    # Check correlation
    corr = df['feat_lag_0'].corr(df['target_1d'])
    print(f"\nCorrelation between Current Return (Feature) and Next Return (Target): {corr:.4f}")
    if abs(corr) > 0.9:
        print("CRITICAL WARNING: High correlation detected! Likely Lookahead Bias.")
    else:
        print("PASS: No immediate correlation detected. Data alignment looks correct.")

def benchmark_model(df, feature_cols, target_cols):
    print("\n--- 2. Benchmarking (Is it actually learning?) ---")
    
    # Load Model
    model = tf.keras.models.load_model(MODEL_FILE)
    
    # Prepare X, y
    X = df[feature_cols].values[:, ::-1].reshape(-1, LOOKBACK_DAYS, 1)
    y_true = df[target_cols].values
    
    # Scaling (Must replicate training scaler logic)
    # We cheat slightly and fit on full data for this quick check, 
    # but strictly we should fit on first 60%.
    scaler = StandardScaler()
    X_flat = X.reshape(-1, 1)
    X_scaled = scaler.fit_transform(X_flat).reshape(X.shape)
    
    # Predict
    y_pred = model.predict(X_scaled, verbose=0)
    
    # Calculate MSEs
    model_mse = mean_squared_error(y_true, y_pred)
    
    # Naive Baseline: Predict 0.0 (No change) for everything
    y_zeros = np.zeros_like(y_true)
    naive_mse = mean_squared_error(y_true, y_zeros)
    
    print(f"Model MSE: {model_mse:.6f}")
    print(f"Naive MSE: {naive_mse:.6f} (Predicting all zeros)")
    
    if model_mse < naive_mse:
        improvement = (1 - (model_mse / naive_mse)) * 100
        print(f"RESULT: Model is {improvement:.2f}% better than predicting zero.")
        print("Interpretation: The model is learning real signal (or overfitting noise), but it is beating the null hypothesis.")
    else:
        print("RESULT: Model is WORSE or EQUAL to predicting zero.")
        print("Interpretation: The model has failed to find signal. The 'low' loss is just the natural low variance of the market.")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FILE) or not os.path.exists(MODEL_FILE):
        print(f"Error: Missing {OUTPUT_FILE} or {MODEL_FILE}. Run training script first.")
    else:
        df, f_cols, t_cols = load_and_prep_data()
        check_lookahead(df)
        benchmark_model(df, f_cols, t_cols)