import pandas as pd
import numpy as np
import gdown
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from flask import Flask, render_template_string
import io
import base64
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Download dataset
print("Downloading dataset...")
url = "https://drive.google.com/uc?id=1ZfDrxUiqlScHYpDetZJXniPRRPit-3Vs"
output = "trading_data.csv"
gdown.download(url, output, quiet=False)

# Load data
print("\n" + "="*70)
print("LOADING AND PREPARING DATA")
print("="*70)
df = pd.read_csv(output)
print(f"✓ Dataset shape: {df.shape}")
print(f"✓ Columns: {df.columns.tolist()}")

# Calculate returns for OHLCV
print("\n[1/6] Calculating returns...")
ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
for col in tqdm(ohlcv_cols, desc="Computing returns"):
    df[f'{col}_return'] = df[col].pct_change()

# Feature Engineering - Technical Indicators
print("\n[2/6] Engineering advanced features...")

features_progress = tqdm(total=8, desc="Creating features")

# Rolling volatility (5, 10, 20 day windows)
for window in [5, 10, 20]:
    df[f'volatility_{window}'] = df['close_return'].rolling(window=window).std()
features_progress.update(1)

# Momentum indicators (Rate of Change)
for window in [5, 10, 20]:
    df[f'momentum_{window}'] = df['close'].pct_change(periods=window)
features_progress.update(1)

# Moving average of returns
for window in [5, 10, 20]:
    df[f'ma_return_{window}'] = df['close_return'].rolling(window=window).mean()
features_progress.update(1)

# Exponential moving averages
for window in [5, 10, 20]:
    df[f'ema_return_{window}'] = df['close_return'].ewm(span=window).mean()
features_progress.update(1)

# Volume-price divergence
for window in [5, 10, 20]:
    df[f'volume_price_corr_{window}'] = df['close_return'].rolling(window).corr(df['volume_return'].rolling(window).mean())
features_progress.update(1)

# High-Low spread (volatility measure)
df['hl_spread'] = (df['high'] - df['low']) / df['close']
for window in [5, 10]:
    df[f'hl_spread_ma_{window}'] = df['hl_spread'].rolling(window).mean()
features_progress.update(1)

# Volume ratio (current vs moving average)
for window in [5, 10, 20]:
    df[f'volume_ratio_{window}'] = df['volume'] / df['volume'].rolling(window).mean()

# RSI-like indicators
for window in [14, 21]:
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
features_progress.update(1)

# Bollinger Band indicators
for window in [20]:
    rolling_mean = df['close'].rolling(window).mean()
    rolling_std = df['close'].rolling(window).std()
    df[f'bb_upper_{window}'] = (df['close'] - rolling_mean) / rolling_std
    df[f'bb_lower_{window}'] = (rolling_mean - df['close']) / rolling_std
features_progress.update(1)

features_progress.close()

# Create lagged features
print("\n[3/6] Creating lagged features (this may take a moment)...")
lag_days = 30
feature_cols = []

# Lagged OHLCV returns with progress bar
for col in tqdm(ohlcv_cols, desc="Lagging OHLCV"):
    for lag in range(1, lag_days + 1):
        lag_col = f'{col}_return_lag_{lag}'
        df[lag_col] = df[f'{col}_return'].shift(lag)
        feature_cols.append(lag_col)

# Lagged technical indicators
tech_indicators = [c for c in df.columns if any(x in c for x in 
    ['volatility', 'momentum', 'ma_return', 'ema_return', 'volume_price_corr', 
     'hl_spread', 'volume_ratio', 'rsi', 'bb_'])]

for col in tqdm(tech_indicators, desc="Lagging indicators"):
    for lag in range(1, 11):
        lag_col = f'{col}_lag_{lag}'
        df[lag_col] = df[col].shift(lag)
        feature_cols.append(lag_col)

# Remove NaN rows
print(f"\n[4/6] Cleaning data...")
print(f"  Rows before removing NaN: {len(df)}")
df = df.dropna()
print(f"  Rows after removing NaN: {len(df)}")
print(f"  ✓ Total features engineered: {len(feature_cols)}")

# Prepare features and target
X = df[feature_cols].values
y = df['perfect_position'].values

# Train/test split (70/30 time-series split)
split_idx = int(len(X) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\n  Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"  Class distribution in train: {np.bincount(y_train.astype(int))}")
print(f"  Class distribution in test: {np.bincount(y_test.astype(int))}")

# Scale features
print("\n[5/6] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training models with progress tracking
print("\n" + "="*70)
print("TRAINING DEEP MODELS (High Complexity)")
print("="*70)

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=3)

# 1. Deep Neural Network with many layers
print("\n[1/5] Training Deep Neural Network (5 layers, 500+ neurons)...")
deep_nn_configs = [
    {'hidden_layer_sizes': (500, 250, 125, 64, 32), 'alpha': 0.0001, 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (400, 200, 100, 50), 'alpha': 0.0001, 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (300, 200, 100, 50, 25), 'alpha': 0.00001, 'learning_rate_init': 0.001},
]

best_deep_nn_score = 0
best_deep_nn_params = None

for params in tqdm(deep_nn_configs, desc="Tuning Deep NN"):
    scores = []
    for train_idx, val_idx in tscv.split(X_train_scaled):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = MLPClassifier(**params, max_iter=2000, early_stopping=True, 
                             validation_fraction=0.15, random_state=42, 
                             batch_size=128, verbose=False)
        model.fit(X_tr, y_tr)
        scores.append(accuracy_score(y_val, model.predict(X_val)))
    
    mean_score = np.mean(scores)
    if mean_score > best_deep_nn_score:
        best_deep_nn_score = mean_score
        best_deep_nn_params = params

print(f"  ✓ Best Deep NN params: {best_deep_nn_params}")
print(f"  ✓ CV Score: {best_deep_nn_score:.4f}")

# 2. Very Deep Random Forest
print("\n[2/5] Training Large Random Forest (500 trees)...")
large_rf_configs = [
    {'n_estimators': 500, 'max_depth': 30, 'min_samples_split': 5, 'min_samples_leaf': 2},
    {'n_estimators': 300, 'max_depth': 40, 'min_samples_split': 3, 'min_samples_leaf': 1},
    {'n_estimators': 400, 'max_depth': 35, 'min_samples_split': 4, 'min_samples_leaf': 2},
]

best_large_rf_score = 0
best_large_rf_params = None

for params in tqdm(large_rf_configs, desc="Tuning Large RF"):
    scores = []
    for train_idx, val_idx in tscv.split(X_train_scaled):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr)
        scores.append(accuracy_score(y_val, model.predict(X_val)))
    
    mean_score = np.mean(scores)
    if mean_score > best_large_rf_score:
        best_large_rf_score = mean_score
        best_large_rf_params = params

print(f"  ✓ Best Large RF params: {best_large_rf_params}")
print(f"  ✓ CV Score: {best_large_rf_score:.4f}")

# 3. Gradient Boosting (complex)
print("\n[3/5] Training Gradient Boosting...")
gb_configs = [
    {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.05, 'subsample': 0.8},
    {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.9},
    {'n_estimators': 400, 'max_depth': 6, 'learning_rate': 0.04, 'subsample': 0.85},
]

best_gb_score = 0
best_gb_params = None

for params in tqdm(gb_configs, desc="Tuning Gradient Boost"):
    scores = []
    for train_idx, val_idx in tscv.split(X_train_scaled):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_tr, y_tr)
        scores.append(accuracy_score(y_val, model.predict(X_val)))
    
    mean_score = np.mean(scores)
    if mean_score > best_gb_score:
        best_gb_score = mean_score
        best_gb_params = params

print(f"  ✓ Best GB params: {best_gb_params}")
print(f"  ✓ CV Score: {best_gb_score:.4f}")

# 4. Wide Neural Network
print("\n[4/5] Training Wide Neural Network (fewer layers, more neurons)...")
wide_nn_configs = [
    {'hidden_layer_sizes': (1000, 500), 'alpha': 0.0001, 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (800, 400), 'alpha': 0.0001, 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (1200, 600, 300), 'alpha': 0.00001, 'learning_rate_init': 0.001},
]

best_wide_nn_score = 0
best_wide_nn_params = None

for params in tqdm(wide_nn_configs, desc="Tuning Wide NN"):
    scores = []
    for train_idx, val_idx in tscv.split(X_train_scaled):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = MLPClassifier(**params, max_iter=2000, early_stopping=True, 
                             validation_fraction=0.15, random_state=42, 
                             batch_size=128, verbose=False)
        model.fit(X_tr, y_tr)
        scores.append(accuracy_score(y_val, model.predict(X_val)))
    
    mean_score = np.mean(scores)
    if mean_score > best_wide_nn_score:
        best_wide_nn_score = mean_score
        best_wide_nn_params = params

print(f"  ✓ Best Wide NN params: {best_wide_nn_params}")
print(f"  ✓ CV Score: {best_wide_nn_score:.4f}")

# 5. Naive Bayes (baseline)
print("\n[5/5] Training Naive Bayes (baseline)...")
best_nb_params = 1e-7

# Train all final models
print("\n" + "="*70)
print("TRAINING FINAL MODELS")
print("="*70)

models_to_train = [
    ("Deep NN", MLPClassifier(**best_deep_nn_params, max_iter=2000, early_stopping=True, 
                              validation_fraction=0.15, random_state=42, batch_size=128, verbose=False)),
    ("Large RF", RandomForestClassifier(**best_large_rf_params, random_state=42, n_jobs=-1)),
    ("Gradient Boost", GradientBoostingClassifier(**best_gb_params, random_state=42)),
    ("Wide NN", MLPClassifier(**best_wide_nn_params, max_iter=2000, early_stopping=True, 
                             validation_fraction=0.15, random_state=42, batch_size=128, verbose=False)),
    ("Naive Bayes", GaussianNB(var_smoothing=best_nb_params))
]

trained_models = {}
predictions = {}

for name, model in tqdm(models_to_train, desc="Training final models"):
    model.fit(X_train_scaled, y_train)
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    trained_models[name] = model
    predictions[name] = {
        'train': train_pred,
        'test': test_pred,
        'train_acc': accuracy_score(y_train, train_pred),
        'test_acc': accuracy_score(y_test, test_pred)
    }
    
    print(f"\n{name}:")
    print(f"  Train Accuracy: {predictions[name]['train_acc']:.4f}")
    print(f"  Test Accuracy: {predictions[name]['test_acc']:.4f}")

# Ensemble Model
print("\n" + "="*70)
print("CREATING ENSEMBLE MODEL")
print("="*70)

ensemble = VotingClassifier(
    estimators=[
        ('deep_nn', trained_models['Deep NN']),
        ('large_rf', trained_models['Large RF']),
        ('gb', trained_models['Gradient Boost']),
        ('wide_nn', trained_models['Wide NN'])
    ],
    voting='hard'
)

print("Training ensemble (this may take a moment)...")
ensemble.fit(X_train_scaled, y_train)
ensemble_train_pred = ensemble.predict(X_train_scaled)
ensemble_test_pred = ensemble.predict(X_test_scaled)

predictions['Ensemble'] = {
    'train': ensemble_train_pred,
    'test': ensemble_test_pred,
    'train_acc': accuracy_score(y_train, ensemble_train_pred),
    'test_acc': accuracy_score(y_test, ensemble_test_pred)
}

print(f"\nEnsemble Model:")
print(f"  Train Accuracy: {predictions['Ensemble']['train_acc']:.4f}")
print(f"  Test Accuracy: {predictions['Ensemble']['test_acc']:.4f}")

# Get price data
price_data = df['close'].values
train_prices = price_data[:split_idx]
test_prices = price_data[split_idx:]

# Trading simulation
def simulate_trading(predictions, actual_prices, starting_capital=1000):
    capital = starting_capital
    capital_history = [capital]
    position = 2
    
    for i in range(1, len(predictions)):
        price_change = (actual_prices[i] - actual_prices[i-1]) / actual_prices[i-1]
        
        if position == 1:
            capital *= (1 + price_change)
        elif position == 0:
            capital *= (1 - price_change)
        
        capital_history.append(capital)
        position = predictions[i]
    
    return np.array(capital_history)

print("\n" + "="*70)
print("SIMULATING TRADING STRATEGIES")
print("="*70)

capital_results = {}
for name in tqdm(predictions.keys(), desc="Simulating trades"):
    capital_results[name] = {
        'train': simulate_trading(predictions[name]['train'], train_prices),
        'test': simulate_trading(predictions[name]['test'], test_prices)
    }

perfect_train_capital = simulate_trading(y_train, train_prices)
perfect_test_capital = simulate_trading(y_test, test_prices)

# Calculate returns (fix the math!)
def calculate_return(final_capital, initial_capital=1000):
    return ((final_capital - initial_capital) / initial_capital) * 100

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

print("\nTraining Performance:")
for name in predictions.keys():
    final = capital_results[name]['train'][-1]
    ret = calculate_return(final)
    print(f"  {name}: ${final:.2f} ({ret:+.1f}%)")
print(f"  Perfect Strategy: ${perfect_train_capital[-1]:.2f} ({calculate_return(perfect_train_capital[-1]):+.1f}%)")

print("\nTest Performance:")
for name in predictions.keys():
    final = capital_results[name]['test'][-1]
    ret = calculate_return(final)
    pct_of_perfect = (final / perfect_test_capital[-1]) * 100
    print(f"  {name}: ${final:.2f} ({ret:+.1f}%) - {pct_of_perfect:.1f}% of perfect")
print(f"  Perfect Strategy: ${perfect_test_capital[-1]:.2f} ({calculate_return(perfect_test_capital[-1]):+.1f}%)")

# Visualization
def create_plots():
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # Training predictions
    ax1 = fig.add_subplot(gs[0, 0])
    for name in ['Deep NN', 'Large RF', 'Ensemble']:
        ax1.plot(predictions[name]['train'], label=name, alpha=0.7)
    ax1.plot(y_train, label='Perfect', linewidth=2, alpha=0.8)
    ax1.set_title('Training: Model Predictions vs Perfect', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test predictions
    ax2 = fig.add_subplot(gs[0, 1])
    for name in ['Deep NN', 'Large RF', 'Ensemble']:
        ax2.plot(predictions[name]['test'], label=name, alpha=0.7)
    ax2.plot(y_test, label='Perfect', linewidth=2, alpha=0.8)
    ax2.set_title('Test: Model Predictions vs Perfect', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Training price
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(train_prices, color='black', linewidth=1)
    ax3.set_title('Training: Price Movement', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Price')
    ax3.grid(True, alpha=0.3)
    
    # Test price
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(test_prices, color='black', linewidth=1)
    ax4.set_title('Test: Price Movement', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Price')
    ax4.grid(True, alpha=0.3)
    
    # Training capital
    ax5 = fig.add_subplot(gs[2, 0])
    for name in capital_results.keys():
        ax5.plot(capital_results[name]['train'], label=name, alpha=0.7)
    ax5.plot(perfect_train_capital, label='Perfect', linewidth=2, color='gold')
    ax5.set_title('Training: Capital Over Time', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Capital ($)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Test capital
    ax6 = fig.add_subplot(gs[2, 1])
    for name in capital_results.keys():
        ax6.plot(capital_results[name]['test'], label=name, alpha=0.7)
    ax6.plot(perfect_test_capital, label='Perfect', linewidth=2, color='gold')
    ax6.set_title('Test: Capital Over Time', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Capital ($)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Accuracy comparison
    ax7 = fig.add_subplot(gs[3, :])
    model_names = list(predictions.keys())
    train_accs = [predictions[name]['train_acc'] for name in model_names]
    test_accs = [predictions[name]['test_acc'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax7.bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
    ax7.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
    ax7.set_ylabel('Accuracy')
    ax7.set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(model_names, rotation=15, ha='right')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Deep Learning Trading Models - High Complexity', fontsize=16, fontweight='bold', y=0.995)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()
    
    return plot_data

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    plot_data = create_plots()
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Deep Learning Trading Models</title>
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            h1 { color: white; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
            .container { max-width: 1600px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 40px rgba(0,0,0,0.3); }
            .info { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 25px; }
            .metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 25px; margin-bottom: 30px; }
            .metric-box { background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #667eea; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            .metric-box h3 { margin-top: 0; color: #667eea; font-size: 1.3em; }
            .metric-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #e0e0e0; }
            .metric-row:last-child { border-bottom: none; }
            .positive { color: #28a745; font-weight: bold; }
            .negative { color: #dc3545; font-weight: bold; }
            .perfect { color: #ffc107; font-weight: bold; }
            img { width: 100%; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        </style>
    </head>
    <body>
        <h1>🚀 Deep Learning Trading Models Dashboard</h1>
        <div class="container">
            <div class="info">
                <strong>🧠 Model Architecture:</strong> Deep NN (5 layers, 500+ neurons), Large RF (500 trees), Gradient Boosting (400 estimators), Wide NN (1000+ neurons), Ensemble Voting
                <br><strong>📊 Features:</strong> {{ total_features }} engineered features (OHLCV returns + 30 lags, volatility, momentum, RSI, Bollinger Bands, volume ratios)
            </div>
            
            <div class="metrics">
                <div class="metric-box">
                    <h3>📈 Training Performance</h3>
                    {% for name, data in train_results.items() %}
                    <div class="metric-row">
                        <span>{{ name }}:</span>
                        <span class="{{ 'positive' if data['return'] > 0 else 'negative' }}">${{ "%.2f"|format(data['final']) }} ({{ "%+.1f"|format(data['return']) }}%)</span>
                    </div>
                    {% endfor %}
                    <div class="metric-row">
                        <span><strong>Perfect Strategy:</strong></span>
                        <span class="perfect">${{ "%.2f"|format(perfect_train_final) }} ({{ "%+.1f"|format(perfect_train_return) }}%)</span>
                    </div>
                </div>
                
                <div class="metric-box">
                    <h3>🎯 Test Performance</h3>
                    {% for name, data in test_results.items() %}
                    <div class="metric-row">
                        <span>{{ name }}:</span>
                        <span class="{{ 'positive' if data['return'] > 0 else 'negative' }}">${{ "%.2f"|format(data['final']) }} ({{ "%+.1f"|format(data['return']) }}%) - {{ "%.1f"|format(data['pct_perfect']) }}%</span>
                    </div>
                    {% endfor %}
                    <div class="metric-row">
                        <span><strong>Perfect Strategy:</strong></span>
                        <span class="perfect">${{ "%.2f"|format(perfect_test_final) }} ({{ "%+.1f"|format(perfect_test_return) }}%)</span>
                    </div>
                </div>
            </div>
            
            <img src="data:image/png;base64,{{ plot_data }}" alt="Trading Analysis">
        </div>
    </body>
    </html>
    """
    
    train_results = {}
    test_results = {}
    
    for name in capital_results.keys():
        train_final = capital_results[name]['train'][-1]
        test_final = capital_results[name]['test'][-1]
        
        train_results[name] = {
            'final': train_final,
            'return': calculate_return(train_final)
        }
        
        test_results[name] = {
            'final': test_final,
            'return': calculate_return(test_final),
            'pct_perfect': (test_final / perfect_test_capital[-1]) * 100
        }
    
    return render_template_string(
        html,
        plot_data=plot_data,
        train_results=train_results,
        test_results=test_results,
        perfect_train_final=perfect_train_capital[-1],
        perfect_train_return=calculate_return(perfect_train_capital[-1]),
        perfect_test_final=perfect_test_capital[-1],
        perfect_test_return=calculate_return(perfect_test_capital[-1]),
        total_features=len(feature_cols)
