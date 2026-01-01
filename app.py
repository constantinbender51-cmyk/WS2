import os
import io
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, send_file, Response

# --- Configuration & Setup ---
matplotlib.use('Agg') 

app = Flask(__name__)

class GlobalState:
    def __init__(self):
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_loss = 0.0
        self.latest_plot_buffer = None
        self.training_finished = False
        self.lock = threading.Lock()

state = GlobalState()

# --- Model Definition ---
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def get_best_fit_slope(y_window):
    """Calculates the slope of the line of best fit for 10 points."""
    # x values are just 0, 1, 2, ..., 9
    n = len(y_window)
    x = np.arange(n)
    # Simple linear regression slope formula
    m = (n * np.sum(x * y_window) - np.sum(x) * np.sum(y_window)) / (n * np.sum(x**2) - (np.sum(x))**2)
    return m

# --- Training Logic ---
def train_model():
    print("--- Starting Training with Best-Fit Slope Features ---")
    
    # 1. Data Generation
    N = 1200
    t = np.linspace(0, 100, N)
    raw_data = np.sin(t) + 0.15 * np.random.normal(size=N)
    
    # 2. Calculate Rolling Best-Fit Slopes
    # Each 'slope' is calculated from the previous 10 amplitude points
    slope_window = 10
    slopes = []
    for i in range(slope_window, len(raw_data)):
        window = raw_data[i - slope_window : i]
        slopes.append(get_best_fit_slope(window))
    
    slopes = np.array(slopes)
    
    # 3. Prepare windowed data for LSTM
    # We use a sequence of 20 'slopes' to predict the current amplitude
    lstm_seq_length = 20
    X, y = [], []
    
    # The 'slopes' array starts representing data from index 'slope_window' onwards
    for i in range(len(slopes) - lstm_seq_length):
        X.append(slopes[i : i + lstm_seq_length])
        # We predict the amplitude at the point corresponding to the end of this sequence
        y.append(raw_data[i + slope_window + lstm_seq_length])
    
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1) 
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)
    
    model = SimpleLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    epochs = 600
    with state.lock:
        state.total_epochs = epochs

    for epoch in range(epochs):
        model.train()
        outputs = model(X)
        loss = criterion(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with state.lock:
            state.current_epoch = epoch + 1
            state.current_loss = loss.item()

        if epoch % 25 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            generate_plot(model, X, y, t, slope_window + lstm_seq_length, loss.item(), epoch, slopes)
            
        time.sleep(0.01)

    with state.lock:
        state.training_finished = True

def generate_plot(model, X, y, t, offset, loss, epoch, slopes):
    model.eval()
    with torch.no_grad():
        predictions = model(X).numpy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # Prediction Plot
    plot_t = t[offset : offset + len(predictions)]
    ax1.plot(plot_t, y.numpy(), label='Actual Amplitude', alpha=0.3, color='teal')
    ax1.plot(plot_t, predictions, label='Predicted (from 10pt Slopes)', color='darkorange', linewidth=2)
    ax1.set_title(f'Best-Fit Slope LSTM | Epoch {epoch} | Loss: {loss:.6f}')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Slope Feature Plot
    ax2.plot(plot_t, slopes[:len(predictions)], label='Input: 10-pt Best-Fit Slope', color='purple', alpha=0.7)
    ax2.set_ylabel('Slope Value')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    with state.lock:
        state.latest_plot_buffer = buf

@app.route('/')
def dashboard():
    with state.lock:
        status = "Completed" if state.training_finished else "Processing..."
        epoch_str = f"{state.current_epoch} / {state.total_epochs}"
        loss_val = f"{state.current_loss:.6f}"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Regression Slope LSTM</title>
        <meta http-equiv="refresh" content="2">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f0f2f5; margin: 0; padding: 40px; color: #1c1e21; }}
            .container {{ background: white; max-width: 1000px; margin: 0 auto; padding: 30px; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.1); }}
            .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #f0f2f5; padding-bottom: 20px; margin-bottom: 20px; }}
            .stat-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 30px; }}
            .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef; }}
            .stat-label {{ font-size: 12px; text-transform: uppercase; color: #65676b; font-weight: 600; }}
            .stat-value {{ font-size: 24px; font-weight: bold; color: #1877f2; }}
            img {{ width: 100%; height: auto; border-radius: 8px; }}
            .badge {{ background: #e7f3ff; color: #1877f2; padding: 4px 12px; border-radius: 20px; font-size: 14px; font-weight: 600; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1 style="margin:0">LSTM: Linear Regression Slopes</h1>
                <span class="badge">{status}</span>
            </div>
            <div class="stat-grid">
                <div class="stat-card"><div class="stat-label">Progress</div><div class="stat-value">{epoch_str}</div></div>
                <div class="stat-card"><div class="stat-label">Current MSE Loss</div><div class="stat-value">{loss_val}</div></div>
                <div class="stat-card"><div class="stat-label">Slope Window</div><div class="stat-value">10 pts</div></div>
            </div>
            <img src="/plot.png?t={time.time()}" />
            <p style="text-align:center; color:#65676b; font-size:14px; margin-top:20px;">
                Inputs: Sequence of 20 slopes (each calculated via Linear Regression on 10 points). Target: Next amplitude.
            </p>
        </div>
    </body>
    </html>
    """

@app.route('/plot.png')
def plot_image():
    buf = None
    with state.lock:
        if state.latest_plot_buffer:
            buf = io.BytesIO(state.latest_plot_buffer.getvalue())
    if buf:
        return send_file(buf, mimetype='image/png')
    return Response("Loading...", status=503)

if __name__ == "__main__":
    training_thread = threading.Thread(target=train_model, daemon=True)
    training_thread.start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)