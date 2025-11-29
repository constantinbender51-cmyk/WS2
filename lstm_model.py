import gdown
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l1_l2
from flask import Flask, render_template_string
import threading
import time
import json

# Global variables for training progress and data
training_progress = {
    'status': 'not_started',  # 'not_started', 'running', 'completed'
    'current_epoch': 0,
    'total_epochs': 40,
    'train_loss': [],
    'val_loss': [],
    'train_predictions': [],
    'train_actual': [],
    'test_predictions': [],
    'test_actual': []
}

app = Flask(__name__)

# HTML template with auto-refresh and charts
html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Training Progress</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <meta http-equiv="refresh" content="60">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .progress { margin: 10px 0; }
        .bar { width: 100%; background-color: #f0f0f0; border-radius: 5px; overflow: hidden; }
        .fill { height: 20px; background-color: #4CAF50; width: 0%; transition: width 0.5s; }
    </style>
</head>
<body>
    <h1>LSTM Model Training Progress</h1>
    <div class="progress">
        <p>Status: <span id="status">{{ status }}</span></p>
        <p>Epoch: <span id="current_epoch">{{ current_epoch }}</span> / <span id="total_epochs">{{ total_epochs }}</span></p>
        <div class="bar">
            <div class="fill" id="epoch_progress" style="width: {{ progress_percent }}%;"></div>
        </div>
    </div>
    <div id="charts" style="display: {{ charts_display }};">
        <h2>Loss Over Epochs</h2>
        <canvas id="lossChart" width="400" height="200"></canvas>
        <h2>Training Predictions vs Actual</h2>
        <canvas id="trainPredictionChart" width="400" height="200"></canvas>
        <h2>Test Predictions vs Actual</h2>
        <canvas id="testPredictionChart" width="400" height="200"></canvas>
    </div>
    <script>
        function updateProgress() {
            fetch('/progress')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status').textContent = data.status;
                    document.getElementById('current_epoch').textContent = data.current_epoch;
                    document.getElementById('total_epochs').textContent = data.total_epochs;
                    const progressPercent = (data.current_epoch / data.total_epochs) * 100;
                    document.getElementById('epoch_progress').style.width = progressPercent + '%';
                    
                    if (data.status === 'running' || data.status === 'completed') {
                        document.getElementById('charts').style.display = 'block';
                        updateCharts(data);
                    } else {
                        document.getElementById('charts').style.display = 'none';
                    }
                })
                .catch(error => console.error('Error fetching progress:', error));
        }
        
        function updateCharts(data) {
            // Loss chart
            const lossCtx = document.getElementById('lossChart').getContext('2d');
            // Destroy existing chart if it exists to avoid duplicates
            if (window.lossChartInstance) {
                window.lossChartInstance.destroy();
            }
            window.lossChartInstance = new Chart(lossCtx, {
                type: 'line',
                data: {
                    labels: Array.from({length: data.train_loss.length}, (_, i) => i + 1),
                    datasets: [
                        { label: 'Training Loss', data: data.train_loss, borderColor: 'blue', fill: false },
                        { label: 'Validation Loss', data: data.val_loss, borderColor: 'red', fill: false }
                    ]
                },
                options: { responsive: true, scales: { y: { beginAtZero: true } } }
            });
            
            // Training Prediction chart
            const trainPredCtx = document.getElementById('trainPredictionChart').getContext('2d');
            if (window.trainPredictionChartInstance) {
                window.trainPredictionChartInstance.destroy();
            }
            window.trainPredictionChartInstance = new Chart(trainPredCtx, {
                type: 'line',
                data: {
                    labels: Array.from({length: data.train_predictions.length}, (_, i) => i),
                    datasets: [
                        { label: 'Training Predictions', data: data.train_predictions, borderColor: 'green', fill: false },
                        { label: 'Training Actual', data: data.train_actual, borderColor: 'orange', fill: false }
                    ]
                },
                options: { responsive: true }
            });
            
            // Test Prediction chart
            const testPredCtx = document.getElementById('testPredictionChart').getContext('2d');
            if (window.testPredictionChartInstance) {
                window.testPredictionChartInstance.destroy();
            }
            window.testPredictionChartInstance = new Chart(testPredCtx, {
                type: 'line',
                data: {
                    labels: Array.from({length: data.test_predictions.length}, (_, i) => i),
                    datasets: [
                        { label: 'Test Predictions', data: data.test_predictions, borderColor: 'blue', fill: false },
                        { label: 'Test Actual', data: data.test_actual, borderColor: 'red', fill: false }
                    ]
                },
                options: { responsive: true }
            });
        }
        
        // Update every second
        setInterval(updateProgress, 1000);
        updateProgress();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    progress_percent = (training_progress['current_epoch'] / training_progress['total_epochs']) * 100 if training_progress['total_epochs'] > 0 else 0
    charts_display = 'block' if training_progress['status'] in ['running', 'completed'] else 'none'
    return render_template_string(html_template, 
                                  status=training_progress['status'],
                                  current_epoch=training_progress['current_epoch'],
                                  total_epochs=training_progress['total_epochs'],
                                  progress_percent=progress_percent,
                                  charts_display=charts_display)

@app.route('/progress')
def progress():
    return json.dumps(training_progress)

def train_model():
    global training_progress
    training_progress['status'] = 'running'
    
    # Download the CSV file from Google Drive
    url = 'https://drive.google.com/uc?id=1QsfDXX4ueu4_IkPKp36EnDHnpZUR19yG'
    output = 'data.csv'
    gdown.download(url, output, quiet=False)

    # Load the CSV data
    data = pd.read_csv(output)
    print(f"Debug: Loaded data with shape {data.shape} and columns {list(data.columns)}")

    # Ensure the data has OHLCV columns and sma_position; adjust column names if necessary
    # Assuming columns: 'date', 'open', 'high', 'low', 'close', 'volume', 'sma_position'
    # If column names differ, update accordingly
    required_columns = ['open', 'high', 'low', 'close', 'volume', 'sma_position']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in the data. Please check the CSV structure.")

    # Prepare features and target
    # Features: past 365 days of close prices for each day i (i-1 to i-365)
    # Target: sma_position for day i
    close_prices = data['close'].values
    sma_positions = data['sma_position'].values

    # Calculate 120-day SMA and handle NaN values
    data['sma_120'] = data['close'].rolling(window=120).mean()
    print(f"Debug: Calculated SMA_120, NaN count: {data['sma_120'].isna().sum()}")
    
    # Create sequences of 22 days for features (reduced by factor of 2)
    X = []
    y = []
    skipped_count = 0
    for i in range(120, len(close_prices)):
        # Features: close prices and SMA values for past 22 days
        close_features = close_prices[i-22:i]
        sma_features = data['sma_120'].values[i-22:i]
        
        # Skip if any NaN values in the sequence
        if np.any(np.isnan(close_features)) or np.any(np.isnan(sma_features)):
            skipped_count += 1
            continue
            
        # Combine close prices and SMA as features
        combined_features = np.column_stack((close_features, sma_features))
        X.append(combined_features)
        y.append(sma_positions[i])

    print(f"Debug: Created sequences, total skipped due to NaN: {skipped_count}, X length: {len(X)}, y length: {len(y)}")
    X = np.array(X)
    y = np.array(y)
    print(f"Debug: X shape after array conversion: {X.shape}, y shape: {y.shape}")

    # Reshape X for LSTM input: (samples, time steps, features)
    # Now we have 2 features per time step (close price and SMA value)
    X = X.reshape((X.shape[0], X.shape[1], 2))

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Standardize the features
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    scaler.fit(X_train_reshaped)
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

    # Check for NaN values in the data
    if np.any(np.isnan(X_train_scaled)) or np.any(np.isnan(y_train)):
        print("Warning: NaN values detected in training data")
        X_train_scaled = np.nan_to_num(X_train_scaled)
        y_train = np.nan_to_num(y_train)
    
    if np.any(np.isnan(X_test_scaled)) or np.any(np.isnan(y_test)):
        print("Warning: NaN values detected in test data")
        X_test_scaled = np.nan_to_num(X_test_scaled)
        y_test = np.nan_to_num(y_test)

    # Build the LSTM model with reduced complexity and no strong regularization
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train_scaled.shape[1], 2)),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')  # Linear activation for regression
    ])

    # Compile the model with gradient clipping
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    # Custom callback to update progress
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            training_progress['current_epoch'] = epoch + 1
            training_progress['train_loss'].append(logs.get('loss'))
            training_progress['val_loss'].append(logs.get('val_loss'))
            
            # Get training predictions for this epoch
            train_pred = self.model.predict(X_train_scaled, verbose=0)
            train_pred_continuous = train_pred.flatten()
            
            # Store training predictions and actual values
            training_progress['train_predictions'] = train_pred_continuous.tolist()
            training_progress['train_actual'] = y_train.tolist()
            
            time.sleep(0.1)  # Small delay to allow progress updates

    # Train the model
    history = model.fit(X_train_scaled, y_train, batch_size=128, epochs=40, validation_data=(X_test_scaled, y_test), verbose=1, callbacks=[ProgressCallback()])

    # Predict on the test set
    y_pred = model.predict(X_test_scaled)

    # Use raw continuous predictions for sma_position in range [-1, 1]
    y_pred_continuous = y_pred.flatten()

    # Calculate performance metrics using continuous predictions (e.g., MSE, MAE)
    mse = np.mean((y_test - y_pred_continuous) ** 2)
    mae = np.mean(np.abs(y_test - y_pred_continuous))

    # Print metrics
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")

    # Update progress with predictions and test data
    training_progress['test_predictions'] = y_pred_continuous.tolist()
    training_progress['test_actual'] = y_test.tolist()
    training_progress['status'] = 'completed'

    # Save the model if needed
    # model.save('lstm_model.h5')

if __name__ == '__main__':
    # Start training in a background thread
    training_thread = threading.Thread(target=train_model)
    training_thread.daemon = True
    training_thread.start()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=8080, debug=False)