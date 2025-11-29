import pandas as pd
import numpy as np
import gdown
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from flask import Flask, render_template_string
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# Download dataset
print("Downloading dataset...")
url = "https://drive.google.com/uc?id=1ZfDrxUiqlScHYpDetZJXniPRRPit-3Vs"
output = "trading_data.csv"
gdown.download(url, output, quiet=False)

# Load data
print("Loading data...")
df = pd.read_csv(output)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Calculate returns for OHLCV
print("Calculating returns...")
ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
for col in ohlcv_cols:
    df[f'{col}_return'] = df[col].pct_change()

# Feature Engineering - Technical Indicators
print("Engineering features...")

# Rolling volatility (5, 10, 20 day windows)
for window in [5, 10, 20]:
    df[f'volatility_{window}'] = df['close_return'].rolling(window=window).std()

# Momentum indicators (Rate of Change)
for window in [5, 10]:
    df[f'momentum_{window}'] = df['close'].pct_change(periods=window)

# Moving average of returns
for window in [5, 10, 20]:
    df[f'ma_return_{window}'] = df['close_return'].rolling(window=window).mean()

# Volume-price divergence
df['volume_price_corr_5'] = df['close_return'].rolling(5).corr(df['volume_return'].rolling(5).mean())
df['volume_price_corr_10'] = df['close_return'].rolling(10).corr(df['volume_return'].rolling(10).mean())

# High-Low spread (volatility measure)
df['hl_spread'] = (df['high'] - df['low']) / df['close']
df['hl_spread_ma_5'] = df['hl_spread'].rolling(5).mean()

# Volume ratio (current vs moving average)
for window in [5, 10]:
    df[f'volume_ratio_{window}'] = df['volume'] / df['volume'].rolling(window).mean()

# Create 30 days of lagged features for returns
print("Creating lagged features...")
lag_days = 30
feature_cols = []

# Lagged OHLCV returns
for col in ohlcv_cols:
    for lag in range(1, lag_days + 1):
        lag_col = f'{col}_return_lag_{lag}'
        df[lag_col] = df[f'{col}_return'].shift(lag)
        feature_cols.append(lag_col)

# Lagged technical indicators
tech_indicators = [c for c in df.columns if any(x in c for x in ['volatility', 'momentum', 'ma_return', 'volume_price_corr', 'hl_spread', 'volume_ratio'])]
for col in tech_indicators:
    for lag in range(1, 11):  # 10 lags for technical indicators
        lag_col = f'{col}_lag_{lag}'
        df[lag_col] = df[col].shift(lag)
        feature_cols.append(lag_col)

# Remove NaN rows
print(f"Rows before removing NaN: {len(df)}")
df = df.dropna()
print(f"Rows after removing NaN: {len(df)}")
print(f"Total features: {len(feature_cols)}")

# Prepare features and target
X = df[feature_cols].values
y = df['perfect_position'].values

# Train/test split (70/30 time-series split)
split_idx = int(len(X) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Class distribution in train: {np.bincount(y_train.astype(int))}")
print(f"Class distribution in test: {np.bincount(y_test.astype(int))}")

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning with Time Series Cross-Validation
print("\nPerforming hyperparameter tuning...")

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

# Naive Bayes (with var_smoothing tuning)
print("Tuning Naive Bayes...")
best_nb_score = 0
best_nb_params = None
for var_smoothing in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
    scores = []
    for train_idx, val_idx in tscv.split(X_train_scaled):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = GaussianNB(var_smoothing=var_smoothing)
        model.fit(X_tr, y_tr)
        scores.append(accuracy_score(y_val, model.predict(X_val)))
    
    mean_score = np.mean(scores)
    if mean_score > best_nb_score:
        best_nb_score = mean_score
        best_nb_params = var_smoothing

print(f"Best NB var_smoothing: {best_nb_params}, CV Score: {best_nb_score:.4f}")

# Random Forest (with regularization)
print("Tuning Random Forest...")
best_rf_score = 0
best_rf_params = None
rf_configs = [
    {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 20, 'min_samples_leaf': 10},
    {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 15, 'min_samples_leaf': 5},
    {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 10, 'min_samples_leaf': 5},
    {'n_estimators': 150, 'max_depth': 20, 'min_samples_split': 10, 'min_samples_leaf': 3},
]

for params in rf_configs:
    scores = []
    for train_idx, val_idx in tscv.split(X_train_scaled):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr)
        scores.append(accuracy_score(y_val, model.predict(X_val)))
    
    mean_score = np.mean(scores)
    if mean_score > best_rf_score:
        best_rf_score = mean_score
        best_rf_params = params

print(f"Best RF params: {best_rf_params}, CV Score: {best_rf_score:.4f}")

# Neural Network (with regularization and early stopping)
print("Tuning Neural Network...")
best_nn_score = 0
best_nn_params = None
nn_configs = [
    {'hidden_layer_sizes': (50, 25), 'alpha': 0.01},
    {'hidden_layer_sizes': (100, 50), 'alpha': 0.01},
    {'hidden_layer_sizes': (100, 50, 25), 'alpha': 0.01},
    {'hidden_layer_sizes': (150, 75), 'alpha': 0.001},
    {'hidden_layer_sizes': (100, 50), 'alpha': 0.001},
]

for params in nn_configs:
    scores = []
    for train_idx, val_idx in tscv.split(X_train_scaled):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = MLPClassifier(**params, max_iter=1000, early_stopping=True, 
                             validation_fraction=0.1, random_state=42)
        model.fit(X_tr, y_tr)
        scores.append(accuracy_score(y_val, model.predict(X_val)))
    
    mean_score = np.mean(scores)
    if mean_score > best_nn_score:
        best_nn_score = mean_score
        best_nn_params = params

print(f"Best NN params: {best_nn_params}, CV Score: {best_nn_score:.4f}")

# Train final models with best parameters
print("\n" + "="*50)
print("Training final models with best parameters...")
print("="*50)

# Naive Bayes
print("\nTraining Naive Bayes...")
nb_model = GaussianNB(var_smoothing=best_nb_params)
nb_model.fit(X_train_scaled, y_train)
nb_train_pred = nb_model.predict(X_train_scaled)
nb_test_pred = nb_model.predict(X_test_scaled)
print(f"Naive Bayes Train Accuracy: {accuracy_score(y_train, nb_train_pred):.4f}")
print(f"Naive Bayes Test Accuracy: {accuracy_score(y_test, nb_test_pred):.4f}")

# Random Forest
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(**best_rf_params, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_train_pred = rf_model.predict(X_train_scaled)
rf_test_pred = rf_model.predict(X_test_scaled)
print(f"Random Forest Train Accuracy: {accuracy_score(y_train, rf_train_pred):.4f}")
print(f"Random Forest Test Accuracy: {accuracy_score(y_test, rf_test_pred):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Neural Network
print("\nTraining Neural Network...")
nn_model = MLPClassifier(**best_nn_params, max_iter=1000, early_stopping=True,
                        validation_fraction=0.1, random_state=42)
nn_model.fit(X_train_scaled, y_train)
nn_train_pred = nn_model.predict(X_train_scaled)
nn_test_pred = nn_model.predict(X_test_scaled)
print(f"Neural Network Train Accuracy: {accuracy_score(y_train, nn_train_pred):.4f}")
print(f"Neural Network Test Accuracy: {accuracy_score(y_test, nn_test_pred):.4f}")

# Get price data for visualization
price_data = df['close'].values
train_prices = price_data[:split_idx]
test_prices = price_data[split_idx:]

# Trading simulation function
def simulate_trading(predictions, actual_prices, starting_capital=1000):
    capital = starting_capital
    capital_history = [capital]
    position = 2  # Start flat
    
    for i in range(1, len(predictions)):
        price_change = (actual_prices[i] - actual_prices[i-1]) / actual_prices[i-1]
        
        if position == 1:  # Long
            capital *= (1 + price_change)
        elif position == 0:  # Short
            capital *= (1 - price_change)
        # position == 2 (flat): no change
        
        capital_history.append(capital)
        position = predictions[i]
    
    return np.array(capital_history)

# Simulate trading for all models
print("\n" + "="*50)
print("Simulating trading strategies...")
print("="*50)

nb_train_capital = simulate_trading(nb_train_pred, train_prices)
nb_test_capital = simulate_trading(nb_test_pred, test_prices)

rf_train_capital = simulate_trading(rf_train_pred, train_prices)
rf_test_capital = simulate_trading(rf_test_pred, test_prices)

nn_train_capital = simulate_trading(nn_train_pred, train_prices)
nn_test_capital = simulate_trading(nn_test_pred, test_prices)

perfect_train_capital = simulate_trading(y_train, train_prices)
perfect_test_capital = simulate_trading(y_test, test_prices)

print(f"\nFinal Capital (Train):")
print(f"  Naive Bayes: ${nb_train_capital[-1]:.2f} ({(nb_train_capital[-1]/perfect_train_capital[-1]*100):.1f}% of perfect)")
print(f"  Random Forest: ${rf_train_capital[-1]:.2f} ({(rf_train_capital[-1]/perfect_train_capital[-1]*100):.1f}% of perfect)")
print(f"  Neural Network: ${nn_train_capital[-1]:.2f} ({(nn_train_capital[-1]/perfect_train_capital[-1]*100):.1f}% of perfect)")
print(f"  Perfect Strategy: ${perfect_train_capital[-1]:.2f}")

print(f"\nFinal Capital (Test):")
print(f"  Naive Bayes: ${nb_test_capital[-1]:.2f} ({(nb_test_capital[-1]/perfect_test_capital[-1]*100):.1f}% of perfect)")
print(f"  Random Forest: ${rf_test_capital[-1]:.2f} ({(rf_test_capital[-1]/perfect_test_capital[-1]*100):.1f}% of perfect)")
print(f"  Neural Network: ${nn_test_capital[-1]:.2f} ({(nn_test_capital[-1]/perfect_test_capital[-1]*100):.1f}% of perfect)")
print(f"  Perfect Strategy: ${perfect_test_capital[-1]:.2f}")

# Calculate Sharpe-like metrics
def calculate_returns_std(capital_history):
    returns = np.diff(capital_history) / capital_history[:-1]
    return returns.std()

print(f"\nRisk (Std of Returns) - Test:")
print(f"  Naive Bayes: {calculate_returns_std(nb_test_capital):.6f}")
print(f"  Random Forest: {calculate_returns_std(rf_test_capital):.6f}")
print(f"  Neural Network: {calculate_returns_std(nn_test_capital):.6f}")
print(f"  Perfect Strategy: {calculate_returns_std(perfect_test_capital):.6f}")

# Create visualizations
def create_plots():
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Trading Model Comparison (Regularized & Feature-Enhanced)', fontsize=16)
    
    # Training - Predictions vs Perfect Position
    ax = axes[0, 0]
    ax.plot(nb_train_pred, label='Naive Bayes', alpha=0.7)
    ax.plot(rf_train_pred, label='Random Forest', alpha=0.7)
    ax.plot(nn_train_pred, label='Neural Network', alpha=0.7)
    ax.plot(y_train, label='Perfect Position', alpha=0.7, linewidth=2)
    ax.set_title('Training: Predictions vs Perfect Position')
    ax.set_xlabel('Time')
    ax.set_ylabel('Position (0=Short, 1=Long, 2=Flat)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Test - Predictions vs Perfect Position
    ax = axes[0, 1]
    ax.plot(nb_test_pred, label='Naive Bayes', alpha=0.7)
    ax.plot(rf_test_pred, label='Random Forest', alpha=0.7)
    ax.plot(nn_test_pred, label='Neural Network', alpha=0.7)
    ax.plot(y_test, label='Perfect Position', alpha=0.7, linewidth=2)
    ax.set_title('Test: Predictions vs Perfect Position')
    ax.set_xlabel('Time')
    ax.set_ylabel('Position (0=Short, 1=Long, 2=Flat)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training - Price
    ax = axes[1, 0]
    ax.plot(train_prices, label='Price', color='black')
    ax.set_title('Training: Price Movement')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Test - Price
    ax = axes[1, 1]
    ax.plot(test_prices, label='Price', color='black')
    ax.set_title('Test: Price Movement')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training - Capital
    ax = axes[2, 0]
    ax.plot(nb_train_capital, label='Naive Bayes', alpha=0.7)
    ax.plot(rf_train_capital, label='Random Forest', alpha=0.7)
    ax.plot(nn_train_capital, label='Neural Network', alpha=0.7)
    ax.plot(perfect_train_capital, label='Perfect Strategy', linewidth=2, color='gold')
    ax.set_title('Training: Capital Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Capital ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Test - Capital
    ax = axes[2, 1]
    ax.plot(nb_test_capital, label='Naive Bayes', alpha=0.7)
    ax.plot(rf_test_capital, label='Random Forest', alpha=0.7)
    ax.plot(nn_test_capital, label='Neural Network', alpha=0.7)
    ax.plot(perfect_test_capital, label='Perfect Strategy', linewidth=2, color='gold')
    ax.set_title('Test: Capital Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Capital ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert plot to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()
    
    return plot_data

# Flask web server
app = Flask(__name__)

@app.route('/')
def index():
    plot_data = create_plots()
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Model Comparison</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            .metrics {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin-bottom: 30px;
            }
            .metric-box {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #007bff;
            }
            .metric-box h3 {
                margin-top: 0;
                color: #007bff;
            }
            .metric-row {
                display: flex;
                justify-content: space-between;
                padding: 5px 0;
            }
            .percentage {
                color: #28a745;
                font-size: 0.9em;
            }
            img {
                width: 100%;
                border-radius: 5px;
            }
            .info-box {
                background-color: #e7f3ff;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                border-left: 4px solid #007bff;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Trading Model Comparison Dashboard</h1>
            
            <div class="info-box">
                <strong>Improvements Applied:</strong> Regularization, Feature Engineering (volatility, momentum, volume ratios), Time-Series CV for hyperparameter tuning
            </div>
            
            <div class="metrics">
                <div class="metric-box">
                    <h3>Training Performance</h3>
                    <div class="metric-row">
                        <span>Naive Bayes:</span>
                        <strong>${{ "%.2f"|format(nb_train_final) }} <span class="percentage">({{ "%.1f"|format(nb_train_pct) }}%)</span></strong>
                    </div>
                    <div class="metric-row">
                        <span>Random Forest:</span>
                        <strong>${{ "%.2f"|format(rf_train_final) }} <span class="percentage">({{ "%.1f"|format(rf_train_pct) }}%)</span></strong>
                    </div>
                    <div class="metric-row">
                        <span>Neural Network:</span>
                        <strong>${{ "%.2f"|format(nn_train_final) }} <span class="percentage">({{ "%.1f"|format(nn_train_pct) }}%)</span></strong>
                    </div>
                    <div class="metric-row">
                        <span>Perfect Strategy:</span>
                        <strong>${{ "%.2f"|format(perfect_train_final) }}</strong>
                    </div>
                </div>
                
                <div class="metric-box">
                    <h3>Test Performance</h3>
                    <div class="metric-row">
                        <span>Naive Bayes:</span>
                        <strong>${{ "%.2f"|format(nb_test_final) }} <span class="percentage">({{ "%.1f"|format(nb_test_pct) }}%)</span></strong>
                    </div>
                    <div class="metric-row">
                        <span>Random Forest:</span>
                        <strong>${{ "%.2f"|format(rf_test_final) }} <span class="percentage">({{ "%.1f"|format(rf_test_pct) }}%)</span></strong>
                    </div>
                    <div class="metric-row">
                        <span>Neural Network:</span>
                        <strong>${{ "%.2f"|format(nn_test_final) }} <span class="percentage">({{ "%.1f"|format(nn_test_pct) }}%)</span></strong>
                    </div>
                    <div class="metric-row">
                        <span>Perfect Strategy:</span>
                        <strong>${{ "%.2f"|format(perfect_test_final) }}</strong>
                    </div>
                </div>
            </div>
            
            <img src="data:image/png;base64,{{ plot_data }}" alt="Trading Analysis">
        </div>
    </body>
    </html>
    """
    
    return render_template_string(
        html_template,
        plot_data=plot_data,
        nb_train_final=nb_train_capital[-1],
        rf_train_final=rf_train_capital[-1],
        nn_train_final=nn_train_capital[-1],
        perfect_train_final=perfect_train_capital[-1],
        nb_test_final=nb_test_capital[-1],
        rf_test_final=rf_test_capital[-1],
        nn_test_final=nn_test_capital[-1],
        perfect_test_final=perfect_test_capital[-1],
        nb_train_pct=nb_train_capital[-1]/perfect_train_capital[-1]*100,
        rf_train_pct=rf_train_capital[-1]/perfect_train_capital[-1]*100,
        nn_train_pct=nn_train_capital[-1]/perfect_train_capital[-1]*100,
        nb_test_pct=nb_test_capital[-1]/perfect_test_capital[-1]*100,
        rf_test_pct=rf_test_capital[-1]/perfect_test_capital[-1]*100,
        nn_test_pct=nn_test_capital[-1]/perfect_test_capital[-1]*100
    )

    )

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌐 Starting web server on http://0.0.0.0:8080")
    print("="*70 + "\n")
    app.run(host='0.0.0.0', port=8080, debug=False)
