import os
import sys
import time
import gdown
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import StandardScaler

# ==========================================
# Configuration
# ==========================================
FILE_ID = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
OUTPUT_FILE = 'market_data.csv'
MINUTES_PER_DAY = 1440
LOOKBACK_DAYS = 14
TARGET_HORIZONS = [1, 2, 3, 5, 8, 13, 21] 
TRAIN_WINDOW_YEARS = 1

# Hyperparameters
BATCH_SIZE = 2048
EPOCHS = 10
UNITS = 64
DROPOUT = 0.2
L1_L2_REG = 1e-5

# ==========================================
# Utilities
# ==========================================
def delayed_print(message):
    """Prints a message with a 0.1s delay to ensure log ordering on Railway."""
    print(message)
    sys.stdout.flush()
    time.sleep(0.1)

class RailwayLogger(Callback):
    """Custom Keras callback for sequential log output."""
    def on_epoch_begin(self, epoch, logs=None):
        delayed_print(f">>> Starting Epoch {epoch + 1}/{EPOCHS}")
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        delayed_print(f">>> Epoch {epoch + 1} Done | Loss: {loss:.8f} | Val: {val_loss:.8f}")

# ==========================================
# 1. Data Download
# ==========================================
if not os.path.exists(OUTPUT_FILE):
    delayed_print("Downloading dataset using gdown...")
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    gdown.download(url, OUTPUT_FILE, quiet=False)
    delayed_print("Download finished.")
else:
    delayed_print("Dataset found locally.")

# ==========================================
# 2. Loading & Preprocessing
# ==========================================
delayed_print("Loading CSV into memory...")

# Read CSV and explicitly parse timestamp to avoid it being treated as a numeric feature
# The error was caused by the timestamp string remaining in the dataframe or index improperly
df = pd.read_csv(OUTPUT_FILE, usecols=['timestamp', 'close'])

delayed_print("Converting timestamps and sorting...")
# Check if timestamp is numeric (Unix) or string
try:
    if df['timestamp'].dtype == 'O': # Object/String
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
except Exception as e:
    delayed_print(f"Timestamp conversion warning: {e}. Attempting default parsing...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

df.sort_values('timestamp', inplace=True)
df.set_index('timestamp', inplace=True)

# Slicing to 1 year + lookback/lookahead buffer to save memory
total_buffer_days = LOOKBACK_DAYS + max(TARGET_HORIZONS) + 5
rows_to_keep = (TRAIN_WINDOW_YEARS * 365 * MINUTES_PER_DAY) + (total_buffer_days * MINUTES_PER_DAY)
if len(df) > rows_to_keep:
    df = df.iloc[-rows_to_keep:].copy()

delayed_print(f"Data ready. Total rows: {len(df)}")

# ==========================================
# 3. Feature & Target Engineering
# ==========================================
delayed_print("Calculating log returns and sequences...")
# Ensure close is float
df['close'] = df['close'].astype(float)
df['log_price'] = np.log(df['close'])

# Generate 14 Daily Log Return Features (at 1-minute resolution)
feature_cols = []
for i in range(LOOKBACK_DAYS):
    col_name = f'feat_lag_{i}'
    # Log return of the day ending at (t - i days)
    df[col_name] = df['log_price'].shift(i * MINUTES_PER_DAY) - df['log_price'].shift((i + 1) * MINUTES_PER_DAY)
    feature_cols.append(col_name)

# Generate multi-horizon Targets
target_cols = []
for day in TARGET_HORIZONS:
    col_name = f'target_{day}d'
    df[col_name] = df['log_price'].shift(-day * MINUTES_PER_DAY) - df['log_price']
    target_cols.append(col_name)

# Drop rows where we don't have full lookback or lookahead
df.dropna(inplace=True)
delayed_print(f"Samples after shifting: {len(df)}")

# Prepare arrays - explicitly select only feature/target columns to exclude index/strings
X = df[feature_cols].values.astype(np.float32)[:, ::-1] # Reverse to have oldest day first
y = df[target_cols].values.astype(np.float32)

# Reshape for LSTM: (samples, timesteps, features)
X = X.reshape((X.shape[0], LOOKBACK_DAYS, 1))

# Scaling
split_idx = int(len(X) * 0.9)
scaler = StandardScaler()
X_train_flat = X[:split_idx].reshape(-1, 1)
scaler.fit(X_train_flat)

X_scaled = scaler.transform(X.reshape(-1, 1)).reshape(X.shape)

X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# ==========================================
# 4. LSTM Model Definition
# ==========================================
delayed_print("Compiling 3-layer LSTM Model...")

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK_DAYS, 1),
         kernel_regularizer=l1_l2(l1=L1_L2_REG, l2=L1_L2_REG)),
    Dropout(DROPOUT),
    
    LSTM(64, return_sequences=True,
         kernel_regularizer=l1_l2(l1=L1_L2_REG, l2=L1_L2_REG)),
    Dropout(DROPOUT),
    
    LSTM(64, return_sequences=False,
         kernel_regularizer=l1_l2(l1=L1_L2_REG, l2=L1_L2_REG)),
    Dropout(DROPOUT),
    
    Dense(len(TARGET_HORIZONS))
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary(print_fn=delayed_print)

# ==========================================
# 5. Training Loop
# ==========================================
delayed_print("Starting Training Session...")

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0,
    callbacks=[RailwayLogger()]
)

delayed_print("Training complete. Saving results...")
model.save("lstm_ohlcv_model.h5")
delayed_print("Workflow finished successfully.")