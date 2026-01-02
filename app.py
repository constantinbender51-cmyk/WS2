import time
import os
import sys
import gdown
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CONFIGURATION
# ==========================================
# FORCE CPU: This prevents silent CUDA driver hangs in containers
DEVICE = torch.device('cpu') 

FILE_ID = '1RreTtTTGZCRcLqk6Ixl85sJAu8izj8DW'
DOWNLOAD_OUTPUT = 'market_data.csv'
SEQ_LENGTH = 60
TRAIN_SPLIT = 0.8
INPUT_DIM = 1
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
NUM_CLASSES = 3
BATCH_SIZE = 32 # Small batch size to ensure it runs
EPOCHS = 10
LEARNING_RATE = 0.001

def slow_print(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    time.sleep(0.1)

# ==========================================
# 2. DATA PREP
# ==========================================
def download_data():
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    if not os.path.exists(DOWNLOAD_OUTPUT):
        slow_print("Downloading CSV...")
        gdown.download(url, DOWNLOAD_OUTPUT, quiet=False, fuzzy=True)
    else:
        slow_print("CSV exists.")

def prepare_data(df):
    slow_print("Calculating returns...")
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    label_map = {-1: 0, 0: 1, 1: 2}
    df['target_class'] = df['signal'].map(label_map)
    
    df.dropna(subset=['log_ret', 'target_class'], inplace=True)
    df['target_class'] = df['target_class'].astype(int)

    data = df['log_ret'].values.reshape(-1, 1)
    labels = df['target_class'].values

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    # Vectorized Windowing (Instant)
    slow_print("Vectorizing sequences...")
    from numpy.lib.stride_tricks import sliding_window_view
    
    # Try/Except for older numpy versions just in case
    try:
        flat_data = data.flatten()
        windows = sliding_window_view(flat_data, window_shape=SEQ_LENGTH)
        X = windows.reshape(*windows.shape, 1)
        # Shift labels to match end of sequence
        y = labels[SEQ_LENGTH:]
        # Trim
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
    except:
        slow_print("Fallback to loop...")
        xs, ys = [], []
        for i in range(len(data) - SEQ_LENGTH):
            xs.append(data[i:(i+SEQ_LENGTH)])
            ys.append(labels[i+SEQ_LENGTH])
        X, y = np.array(xs), np.array(ys)

    return X, y

# ==========================================
# 3. MODEL
# ==========================================
class LSTMClassifier(nn.Module):
    def __init__(self):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, batch_first=True, dropout=DROPOUT)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    slow_print(f"Script starting on {DEVICE}...")
    
    # 1. Download
    download_data()
    
    # 2. Data
    try:
        df = pd.read_csv(DOWNLOAD_OUTPUT)
        X, y = prepare_data(df)
        slow_print(f"Data Ready. X: {X.shape}")
    except Exception as e:
        slow_print(f"Data Error: {e}")
        sys.exit(1)

    # 3. Tensors
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    
    train_size = int(len(X_tensor) * TRAIN_SPLIT)
    train_data = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
    test_data = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])

    # CRITICAL: num_workers=0 avoids multiprocessing deadlocks
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. Model Init
    model = LSTMClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- SANITY CHECK (The "Is it blocked?" Test) ---
    slow_print("Performing Sanity Check (1 Forward Pass)...")
    try:
        # Create dummy input of correct shape (1, SEQ_LENGTH, 1)
        dummy_input = torch.randn(1, SEQ_LENGTH, 1).to(DEVICE)
        dummy_out = model(dummy_input)
        slow_print(f"Sanity Check Passed. Output shape: {dummy_out.shape}")
    except Exception as e:
        slow_print(f"Sanity Check FAILED: {e}")
        sys.exit(1)
    # -----------------------------------------------

    slow_print("Model initialized. Starting training...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Explicit iterator check
        slow_print(f"Starting Epoch {epoch+1} loop...")
        
        batch_count = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Print specifically on the very first batch to prove movement
            if epoch == 0 and batch_count == 1:
                slow_print(">> First batch executed successfully!")
            
            # Print every 100 batches to show life
            if batch_count % 100 == 0:
                slow_print(f"Processing batch {batch_count}...")

        avg_loss = total_loss / len(train_loader)
        slow_print(f"Epoch {epoch+1} Complete. Loss: {avg_loss:.4f}")

    slow_print("Saving model...")
    torch.save(model.state_dict(), 'lstm_model.pth')
    slow_print("Script finished.")