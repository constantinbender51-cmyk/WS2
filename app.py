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
LOOKBACK_DAYS = 14
TARGET_HORIZONS = [1, 2, 3, 5, 8, 13, 21] 

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
UNITS = 64
DROPOUT = 0.2
L1_L2_REG = 1e-4

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
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            delayed_print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {loss:.6f} | Val: {val_loss:.6f}")

# ==========================================
# 1. Data Download
# ==========================================
if not os.path.exists(OUTPUT_FILE):
    delayed_print("Downloading dataset...")
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    gdown.download(url, OUTPUT_FILE, quiet=False)
    delayed_print("Download finished.")

# ==========================================
# 2. Loading & Resampling
# ==========================================
delayed_print("Loading CSV and resampling to Daily...")
df = pd.read_csv(OUTPUT_FILE, usecols=['timestamp', 'close'])

# Robust timestamp parsing
if df['timestamp'].dtype != 'O':
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
else:
    df['timestamp'] = pd.to_datetime(df['timestamp'])

df.sort_values('timestamp', inplace=True)
df.set_index('timestamp', inplace=True)

# Resample to Daily (Last close of the day)
df_daily = df['close'].resample('1D').last().dropna().to_frame()
delayed_print(f"Resampled from ~4M rows to {len(df_daily)} days.")

# ==========================================
# 3. Feature & Target Engineering
# ==========================================
delayed_print("Engineering features (14-day lookback) and targets...")
df_daily['log_price'] = np.log(df_daily['close'])

# Features: 14 sequential daily log returns
feature_cols = []
for i in range(LOOKBACK_DAYS):
    col_name = f'feat_lag_{i}'
    df_daily[col_name] = df_daily['log_price'].shift(i) - df_daily['log_price'].shift(i + 1)
    feature_cols.append(col_name)

# Targets: Multi-day forward log returns
target_cols = []
for day in TARGET_HORIZONS:
    col_name = f'target_{day}d'
    df_daily[col_name] = df_daily['log_price'].shift(-day) - df_daily['log_price']
    target_cols.append(col_name)

df_daily.dropna(inplace=True)

# Prepare arrays
X = df_daily[feature_cols].values[:, ::-1] # Reverse for chronological order
y = df_daily[target_cols].values

# Reshape for LSTM: (samples, timesteps, features)
X = X.reshape((X.shape[0], LOOKBACK_DAYS, 1))

# ==========================================
# 4. Data Split (60/20/20)
# ==========================================
n = len(X)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# Scale based on training data
scaler = StandardScaler()
X_train_shape = X_train.shape
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train_shape)

X_val_scaled = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

delayed_print(f"Split sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ==========================================
# 5. Model Definition
# ==========================================
delayed_print("Building LSTM Model (3 layers, 64 units)...")
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

model.compile(optimizer='adam', loss='mse')

# ==========================================
# 6. Training
# ==========================================
delayed_print("Starting Training...")
model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0,
    callbacks=[RailwayLogger()]
)

# Final Evaluation
test_loss = model.evaluate(X_test_scaled, y_test, verbose=0)
delayed_print(f"Final Test Loss (MSE): {test_loss:.8f}")

model.save("simplified_lstm_model.h5")
delayed_print("Process Complete. Model saved.")