import time
import sys
import os
import gdown
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from numpy.lib.stride_tricks import sliding_window_view
import itertools

# ==========================================
# 1. CONFIG & HYPERPARAMETER GRID
# ==========================================
torch.set_num_threads(12)
DEVICE = torch.device('cpu')

FILE_ID = '1SagUdIPk-9nU1tlNAOqVAyILZV4Mflj0'
DOWNLOAD_OUTPUT = 'market_data.csv'
TRAIN_SPLIT = 0.8
BATCH_SIZE = 4096
EPOCHS_PER_RUN = 30  # As requested: try first 30 epochs
MAX_LR = 0.01
WEIGHT_DECAY = 1e-3

# Define the Grid
param_grid = {
    'seq_length': [20, 30, 60],
    'hidden_dim': [128, 256, 512],
    'dropout': [0.2, 0.4],
    'num_layers': [2]
}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ==========================================
# 2. DATA PIPELINE
# ==========================================
def get_processed_data():
    if not os.path.exists(DOWNLOAD_OUTPUT):
        log(f"Downloading data...")
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, DOWNLOAD_OUTPUT, quiet=True, fuzzy=True)

    df = pd.read_csv(DOWNLOAD_OUTPUT)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    label_map = {-1: 0, 0: 1, 1: 2}
    df['target_class'] = df['signal'].map(label_map)
    df.dropna(subset=['log_ret', 'target_class'], inplace=True)
    
    data_val = df['log_ret'].values.reshape(-1, 1)
    labels_val = df['target_class'].values.astype(int)
    
    scaler = RobustScaler()
    data_val = scaler.fit_transform(data_val)
    
    return data_val, labels_val

def create_sequences(data, labels, seq_length):
    windows = sliding_window_view(data.flatten(), window_shape=seq_length)
    X = windows.reshape(*windows.shape, 1)
    y = labels[seq_length-1:]
    min_len = min(len(X), len(y))
    return X[:min_len], y[:min_len]

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Last time step
        out = self.bn(out)
        out = torch.relu(out)
        out = self.dropout(out)
        return self.fc(out)

# ==========================================
# 4. GRID SEARCH EXECUTION
# ==========================================
if __name__ == "__main__":
    raw_data, raw_labels = get_processed_data()
    
    keys, values = zip(*param_grid.items())
    grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_loss = float('inf')
    best_params = None
    
    log(f"Starting Grid Search: {len(grid_combinations)} combinations found.")

    for i, params in enumerate(grid_combinations):
        log(f"--- Run {i+1}/{len(grid_combinations)}: {params} ---")
        
        # Prepare data for this specific sequence length
        X, y = create_sequences(raw_data, raw_labels, params['seq_length'])
        split_idx = int(len(X) * TRAIN_SPLIT)
        
        X_train = torch.tensor(X[:split_idx], dtype=torch.float32)
        y_train = torch.tensor(y[:split_idx], dtype=torch.long)
        X_test = torch.tensor(X[split_idx:], dtype=torch.float32)
        y_test = torch.tensor(y[split_idx:], dtype=torch.long)

        # Compute weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.numpy())
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

        model = LSTMClassifier(1, params['hidden_dim'], params['num_layers'], params['dropout'], 3).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=MAX_LR, steps_per_epoch=len(train_loader), epochs=EPOCHS_PER_RUN
        )

        min_val_loss_this_run = float('inf')

        for epoch in range(EPOCHS_PER_RUN):
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for bx, by in test_loader:
                    total_val_loss += criterion(model(bx), by).item()
            
            avg_val_loss = total_val_loss / len(test_loader)
            if avg_val_loss < min_val_loss_this_run:
                min_val_loss_this_run = avg_val_loss
        
        log(f"Completed. Best Val Loss for this config: {min_val_loss_this_run:.4f}")

        if min_val_loss_this_run < best_loss:
            best_loss = min_val_loss_this_run
            best_params = params
            log(f"*** NEW BEST CONFIG FOUND ***")

    log("==========================================")
    log("GRID SEARCH COMPLETE")
    log(f"Best Parameters: {best_params}")
    log(f"Minimum Validation Loss: {best_loss:.4f}")
    log("==========================================")