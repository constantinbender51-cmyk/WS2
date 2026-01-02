import time
import sys
import os
import logging
import traceback
import gdown
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from numpy.lib.stride_tricks import sliding_window_view

# ==========================================
# 0. LOGGING SETUP
# ==========================================
# Configure logging to output to stdout immediately (no buffering)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

def log_step(msg):
    """Logs a message and pauses briefly to ensure order/readability."""
    logger.info(msg)
    # Flush stdout manually just to be safe for Railway
    sys.stdout.flush()
    time.sleep(0.1)

def log_error(msg):
    """Logs error with traceback."""
    logger.error(msg)
    logger.error(traceback.format_exc())
    sys.stdout.flush()
    time.sleep(0.1)

# ==========================================
# 1. HYPERPARAMETERS
# ==========================================
# Configuration
FILE_ID = '1RreTtTTGZCRcLqk6Ixl85sJAu8izj8DW'
DOWNLOAD_OUTPUT = 'market_data.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Params
SEQ_LENGTH = 2
TRAIN_SPLIT = 0.1

# Model Params
INPUT_DIM = 1
HIDDEN_DIM = 12
NUM_LAYERS = 2
DROPOUT = 0.2
NUM_CLASSES = 3 

# Training Params
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

# ==========================================
# 2. ROBUST DATA HANDLING
# ==========================================
def download_data():
    log_step(f"Attempting download from Google Drive ID: {FILE_ID}")
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    
    try:
        if os.path.exists(DOWNLOAD_OUTPUT):
            # Check if file is empty
            if os.path.getsize(DOWNLOAD_OUTPUT) > 0:
                log_step("File already exists and is not empty. Skipping download.")
                return
            else:
                log_step("File exists but is empty. Re-downloading...")
        
        gdown.download(url, DOWNLOAD_OUTPUT, quiet=False, fuzzy=True)
        
        if not os.path.exists(DOWNLOAD_OUTPUT):
            raise FileNotFoundError("Download command finished, but file was not created.")
            
        log_step("Download successful.")
        
    except Exception:
        log_error("Failed to download data.")
        sys.exit(1)

def prepare_data(df):
    try:
        log_step("Preprocessing data...")
        
        # Validation 1: Columns
        required_cols = ['close', 'signal']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV missing columns. Found: {df.columns}, Expected: {required_cols}")

        # Validation 2: Signal Content
        unique_signals = df['signal'].unique()
        log_step(f"Unique signals found in CSV: {unique_signals}")
        
        # 1. Calculate Log Returns
        # eps added to avoid log(0) if price matches previous price perfectly
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1).replace(0, np.nan))
        
        # 2. Map Targets
        label_map = {-1: 0, 0: 1, 1: 2}
        df['target_class'] = df['signal'].map(label_map)
        
        # Check for unmapped signals
        if df['target_class'].isnull().sum() > 0:
            logger.warning(f"Found {df['target_class'].isnull().sum()} rows with signals outside -1, 0, 1. Dropping them.")

        # 3. Clean NaNs
        before_drop = len(df)
        df.dropna(subset=['log_ret', 'target_class'], inplace=True)
        df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
        df['target_class'] = df['target_class'].astype(int)
        log_step(f"Dropped {before_drop - len(df)} rows due to NaN/Inf/Invalid Signals.")

        if len(df) < SEQ_LENGTH:
            raise ValueError(f"Not enough data left after cleaning ({len(df)} rows) to form sequences of length {SEQ_LENGTH}")

        # 4. Extract arrays
        data = df['log_ret'].values.reshape(-1, 1)
        labels = df['target_class'].values

        # 5. Normalize
        log_step("Normalizing features (StandardScaler)...")
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        # 6. Vectorized Windowing
        log_step("Creating sequences using vectorized sliding window...")
        flat_data = data.flatten()
        windows = sliding_window_view(flat_data, window_shape=SEQ_LENGTH)
        X = windows.reshape(*windows.shape, 1) # (N, SEQ, 1)
        
        # Align labels: We want to predict the label at the END of the sequence
        # or the one immediately following. Standard approach: predict label at t+1?
        # Based on "Close is input, Signal is target", usually Signal_t corresponds to Close_t
        # If we input Close_{t-N} to Close_t, we usually want to predict Signal_{t+1} or Signal_t.
        # Assuming we want to predict the signal concurrent with the last price in the window:
        y = labels[SEQ_LENGTH - 1:] 
        
        # Adjust lengths
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
        
        log_step(f"Data Prepared. Input shape: {X.shape}, Target shape: {y.shape}")
        return X, y

    except Exception:
        log_error("Error during data preparation.")
        sys.exit(1)

# ==========================================
# 3. MODEL DEFINITION
# ==========================================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x: (batch, seq, feature)
        out, _ = self.lstm(x)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    log_step("========================================")
    log_step("      STARTING ROBUST LSTM TRAINER      ")
    log_step(f"      Device: {DEVICE}")
    log_step("========================================")

    # --- STEP 1: DOWNLOAD ---
    download_data()

    # --- STEP 2: LOAD CSV ---
    try:
        log_step("Reading CSV...")
        df = pd.read_csv(DOWNLOAD_OUTPUT)
        log_step(f"CSV Loaded. Rows: {len(df)}")
    except Exception:
        log_error("Failed to read CSV.")
        sys.exit(1)

    # --- STEP 3: PREPROCESS ---
    X, y = prepare_data(df)

    # --- STEP 4: TENSORS ---
    try:
        log_step("Converting to PyTorch Tensors...")
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).long()
        
        dataset_size = len(X_tensor)
        train_size = int(dataset_size * TRAIN_SPLIT)
        
        train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
        test_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])
        
        log_step(f"Train samples: {len(train_dataset)} | Val samples: {len(test_dataset)}")
        
        # CRITICAL: num_workers=0 prevents deadlocks in simple scripts
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)
    except Exception:
        log_error("Error creating Tensors/DataLoaders.")
        sys.exit(1)

    # --- STEP 5: MODEL INIT ---
    try:
        log_step("Initializing Model...")
        model = LSTMClassifier(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        log_step("Model initialized successfully.")
    except Exception:
        log_error("Error initializing model.")
        sys.exit(1)

    # --- STEP 6: TRAINING LOOP ---
    try:
        log_step("Starting Training Loop...")
        
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            start_time = time.time()
            
            # Progress tracking
            total_batches = len(train_loader)
            
            for i, (bx, by) in enumerate(train_loader):
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(bx)
                loss = criterion(outputs, by)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += by.size(0)
                correct += (predicted == by).sum().item()

                # Detailed logging for the first batch of the first epoch
                # This confirms the loop is actually running
                if epoch == 0 and i == 0:
                    log_step(">>> First batch processed! The loop is ALIVE. <<<")
                
                # Log progress every 25% of the epoch
                if (i + 1) % max(1, (total_batches // 4)) == 0:
                    log_step(f"Epoch {epoch+1} Progress: {i+1}/{total_batches} batches...")

            # End of Epoch Metrics
            train_acc = 100 * correct / total
            avg_loss = total_loss / total_batches
            
            # Validation
            model.eval()
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                for bx, by in test_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    out = model(bx)
                    _, pred = torch.max(out.data, 1)
                    v_total += by.size(0)
                    v_correct += (pred == by).sum().item()
            
            val_acc = 100 * v_correct / v_total if v_total > 0 else 0
            epoch_time = time.time() - start_time
            
            log_step(f"Summary Ep {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Time: {epoch_time:.2f}s")

        log_step("Training complete.")
        
        save_path = 'lstm_model.pth'
        torch.save(model.state_dict(), save_path)
        log_step(f"Model saved to {save_path}")
        
    except KeyboardInterrupt:
        log_step("Training interrupted by user.")
    except Exception:
        log_error("CRITICAL ERROR during training loop.")
        sys.exit(1)

if __name__ == "__main__":
    main()