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
TARGET_HORIZONS = [1, 2, 3, 5, 8, 13, 21] # Days
TRAIN_WINDOW_YEARS = 1

# Hyperparameters
BATCH_SIZE = 1024  # Large batch size for speed on large datasets
EPOCHS = 10
UNITS = 64
DROPOUT = 0.2
REG_FACTOR = 0.001

# ==========================================
# Utilities
# ==========================================
def delayed_print(message):
    """Prints a message with a slight delay to enforce log ordering."""
    print(message)
    sys.stdout.flush()
    time.sleep(0.1)

class RailwayLogger(Callback):
    """Custom Keras callback to ensure logs print cleanly on cloud platforms."""
    def on_epoch_begin(self, epoch, logs=None):
        delayed_print(f"--- Starting Epoch {epoch + 1}/{EPOCHS} ---")

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        delayed_print(f"Epoch {epoch + 1} finished. Loss: {loss:.6f} | Val Loss: {val_loss:.6f}")

# ==========================================
# 1. Data Download
# ==========================================
if not os.path.exists(OUTPUT_FILE):
    delayed_print(f"Downloading file from Google Drive (ID: {FILE_ID})...")
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    gdown.download(url, OUTPUT_FILE, quiet=False)
    delayed_print("Download complete.")
else:
    delayed_print("Dataset already exists. Skipping download.")

# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================
delayed_print("Loading CSV data... (This may take a moment)")

# Reading only necessary columns to save memory
# Assuming standard OHLCV format, we primarily need Close (and maybe Volume)
# We'll read the first few rows to check structure, then read full
df = pd.read_csv(OUTPUT_FILE)

# basic cleanup
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.set_index('Timestamp', inplace=True)
elif 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

# Sort index to ensure time order
df.sort_index(inplace=True)

# Handling NaN
df.dropna(inplace=True)
delayed_print(f"Total rows loaded: {len(df)}")

# ==========================================
# 3. Feature Engineering (Complex)
# ==========================================
# Requirement: "train on 1 year 14 days lookback daily"
# Strategy: Slice the last ~1 year of data + buffer for lookback calculation
required_rows = (TRAIN_WINDOW_YEARS * 365 * MINUTES_PER_DAY) + (25 * MINUTES_PER_DAY) # 25 days buffer
if len(df) > required_rows:
    df = df.iloc[-required_rows:].copy()
    delayed_print(f"Sliced to last 1 year of data: {len(df)} rows")

delayed_print("Generating features...")

# We need log returns. 
# "every minute has 14 sequences of log return"
# This implies at time T, we want: 
# [LogRet(T-14d), LogRet(T-13d), ... LogRet(T-1d)]? 
# Or 14 steps of "daily" candles leading up to T?
# We will construct a feature set where for every minute, we look at the price 
# 1 day ago, 2 days ago... up to 14 days ago, and calculate the daily log return sequence.

# Calculate log price to make subtraction easier
df['log_price'] = np.log(df['Close'])

# Create Features: 14 lags of Daily Returns relative to the specific minute
# Feature k (where k=0..13) = LogPrice(t - k*days) - LogPrice(t - (k+1)*days)
feature_cols = []
for i in range(LOOKBACK_DAYS):
    # The return for the day ending at (t - i days)
    # e.g., if i=0, it's return from t-1d to t
    col_name = f'feat_vol_d{i}'
    
    # Current point in sequence: t - (i * 1440)
    # Previous point in sequence: t - ((i + 1) * 1440)
    shift_current = i * MINUTES_PER_DAY
    shift_prev = (i + 1) * MINUTES_PER_DAY
    
    # We use shift() to align past data to current row. 
    # shift(X) moves data DOWN, so row T gets data from T-X.
    df[col_name] = df['log_price'].shift(shift_current) - df['log_price'].shift(shift_prev)
    feature_cols.append(col_name)

# Create Targets: Forward looking log returns
target_cols = []
for day in TARGET_HORIZONS:
    col_name = f'target_{day}d'
    # Target: Return from t to t + (day * 1440)
    # We use shift(-X) to move future data UP to current row.
    shift_future = -1 * day * MINUTES_PER_DAY
    df[col_name] = df['log_price'].shift(shift_future) - df['log_price']
    target_cols.append(col_name)

# Drop NaNs created by shifting (both lookback and lookahead)
delayed_print("Cleaning NaN values from shifting...")
df.dropna(inplace=True)

delayed_print(f"Final dataset shape for training: {df.shape}")

# ==========================================
# 4. Dataset Creation
# ==========================================
# Input Shape: (Samples, Sequence_Length, Features)
# The user asked for "14 sequences of log return".
# Our columns `feature_cols` [d0, d1, ... d13] represent the sequence.
# We need to reshape this from (N, 14) to (N, 14, 1) for the LSTM.

X = df[feature_cols].values
y = df[target_cols].values

# Reverse columns so d13 (oldest) comes first, d0 (newest) comes last in sequence
X = X[:, ::-1] 

# Reshape for LSTM: (Samples, TimeSteps, Features)
X = X.reshape((X.shape[0], LOOKBACK_DAYS, 1))

# Scaling is crucial for LSTMs, though Log Returns are already small.
# We'll stick to raw log returns as they are naturally scaled around 0, 
# but a scaler is safer for gradients.
scaler = StandardScaler()
# Flatten, scale, reshape back
X_flat = X.reshape(-1, 1)
X_scaled_flat = scaler.fit_transform(X_flat)
X = X_scaled_flat.reshape(X.shape[0], LOOKBACK_DAYS, 1)

delayed_print("Data prepared. Splitting train/test...")

# Time-series split (no shuffling)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

delayed_print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ==========================================
# 5. Model Architecture
# ==========================================
delayed_print("Building LSTM Model...")

model = Sequential()

# Layer 1
model.add(LSTM(UNITS, 
               return_sequences=True, 
               input_shape=(LOOKBACK_DAYS, 1),
               kernel_regularizer=l1_l2(l1=REG_FACTOR, l2=REG_FACTOR),
               recurrent_regularizer=l1_l2(l1=REG_FACTOR, l2=REG_FACTOR)))
model.add(Dropout(DROPOUT))

# Layer 2
model.add(LSTM(UNITS, 
               return_sequences=True,
               kernel_regularizer=l1_l2(l1=REG_FACTOR, l2=REG_FACTOR)))
model.add(Dropout(DROPOUT))

# Layer 3
model.add(LSTM(UNITS, 
               return_sequences=False, # Last LSTM layer
               kernel_regularizer=l1_l2(l1=REG_FACTOR, l2=REG_FACTOR)))
model.add(Dropout(DROPOUT))

# Output Layer
# 7 neurons for the 7 target horizons (1d, 2d, 3d, 5d, 8d, 13d, 21d)
model.add(Dense(len(TARGET_HORIZONS)))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.summary(print_fn=delayed_print)

# ==========================================
# 6. Training
# ==========================================
delayed_print("Starting training...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0, # Turn off default progress bar to use our custom clean logger
    callbacks=[RailwayLogger()]
)

delayed_print("Training complete.")

# ==========================================
# 7. Sample Prediction
# ==========================================
sample_input = X_test[0:1]
prediction = model.predict(sample_input)

delayed_print("\n--- Sample Prediction (Log Returns) ---")
for i, horizon in enumerate(TARGET_HORIZONS):
    delayed_print(f"Horizon {horizon} Days: {prediction[0][i]:.6f}")

delayed_print("\nDone.")