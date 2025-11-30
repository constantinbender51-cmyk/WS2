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
    'total_epochs': 1600,
    'train_loss': [],
    'val_loss': [],
    'train_predictions': [],
    'train_actual': [],
    'test_predictions': [],
    'test_actual': [],
    'data_with_predictions': None
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
    <div id="download_section" style="display: {{ charts_display }};">
        <h2>Download Data</h2>
        <p><a href="/download" id="download_link">Download CSV with OHLCV, SMA Position, and Model Output</a></p>
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
                        document.getElementById('download_section').style.display = 'block';
                        updateCharts(data);
                    } else {
                        document.getElementById('charts').style.display = 'none';
                        document.getElementById('download_section').style.display = 'none';
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
                options: { responsive: true, scales: { y: { type: 'logarithmic', log: true } } }
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

@app.route('/download')
def download_csv():
    if training_progress['status'] != 'completed' or training_progress['data_with_predictions'] is None:
        return "Training not completed or data not available", 400
    data_df = pd.DataFrame(training_progress['data_with_predictions'])
    # Select columns: datetime, OHLCV, sma_position, and model_output
    columns_to_include = ['date', 'open', 'high', 'low', 'close', 'sma_position', 'model_output']
    # Filter to available columns to avoid KeyError
    available_columns = [col for col in columns_to_include if col in data_df.columns]
    data_df = data_df[available_columns]
    # Rename 'date' to 'datetime' for clarity
    data_df = data_df.rename(columns={'date': 'datetime'})
    csv_data = data_df.to_csv(index=False)
    return csv_data, 200, {'Content-Type': 'text/csv', 'Content-Disposition': 'attachment; filename=ohlcv_sma_model_output.csv'}

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

    # Calculate 365-day SMA, 120-day SMA, (close - SMA) / SMA for both, then handle NaN values
    data['sma_365'] = data['close'].rolling(window=365).mean()
    data['sma_120'] = data['close'].rolling(window=120).mean()
    data['close_over_sma_365'] = (data['close'] - data['sma_365']) / data['sma_365']
    data['close_over_sma_120'] = (data['close'] - data['sma_120']) / data['sma_120']
    print(f"Debug: Calculated SMA_365, NaN count: {data['sma_365'].isna().sum()}")
    print(f"Debug: Calculated SMA_120, NaN count: {data['sma_120'].isna().sum()}")
    print(f"Debug: Calculated close / sma_365, NaN count: {data['close_over_sma_365'].isna().sum()}")
    print(f"Debug: Calculated close / sma_120, NaN count: {data['close_over_sma_120'].isna().sum()}")
    
    # Create sequences of 2 days for features
    X = []
    y = []
    skipped_count = 0
    for i in range(365, len(close_prices)):
        # Features: SMA_365, SMA_120, (close - SMA) / SMA for both, values for the past 2 days
        sma_365_features = data['sma_365'].values[i-2:i]
        sma_120_features = data['sma_120'].values[i-2:i]
        close_over_sma_365_features = data['close_over_sma_365'].values[i-2:i]
        close_over_sma_120_features = data['close_over_sma_120'].values[i-2:i]
        
        # Skip if any NaN values in the sequence
        if np.any(np.isnan(sma_365_features)) or np.any(np.isnan(sma_120_features)) or np.any(np.isnan(close_over_sma_365_features)) or np.any(np.isnan(close_over_sma_120_features)):
            skipped_count += 1
            continue
            
        # Combine SMA_365, SMA_120, (close - SMA) / SMA for both, as features
        combined_features = np.column_stack((sma_365_features, sma_120_features, close_over_sma_365_features, close_over_sma_120_features))
        X.append(combined_features)
        y.append(sma_positions[i])

    print(f"Debug: Created sequences, total skipped due to NaN: {skipped_count}, X length: {len(X)}, y length: {len(y)}")
    X = np.array(X)
    y = np.array(y)
    print(f"Debug: X shape after array conversion: {X.shape}, y shape: {y.shape}")

    # Reshape X for LSTM input: (samples, time steps, features)
    # Now we have 4 features per time step (SMA_365, SMA_120, (close - SMA) / SMA for both)
    X = X.reshape((X.shape[0], X.shape[1], 4))

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

    # Build the LSTM model with specified architecture
    model = Sequential([
        LSTM(16, return_sequences=True, input_shape=(X_train_scaled.shape[1], 4), kernel_regularizer=l1_l2(l1=0.0004, l2=0.0002)),
        Dropout(0.6),
        LSTM(8, return_sequences=False, kernel_regularizer=l1_l2(l1=0.0004, l2=0.0002)),
        Dropout(0.6),
        Dense(4, activation='relu', kernel_regularizer=l1_l2(l1=0.0004, l2=0.0002)),
        Dense(1, activation='tanh')  # Tanh activation for bounded predictions [-1, 1]
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
            
            # Get test predictions for this epoch
            test_pred = self.model.predict(X_test_scaled, verbose=0)
            test_pred_continuous = test_pred.flatten()
            
            # Store training and test predictions and actual values
            training_progress['train_predictions'] = train_pred_continuous.tolist()
            training_progress['train_actual'] = y_train.tolist()
            training_progress['test_predictions'] = test_pred_continuous.tolist()
            training_progress['test_actual'] = y_test.tolist()
            
            time.sleep(0.1)  # Small delay to allow progress updates

    # Train the model
    history = model.fit(X_train_scaled, y_train, batch_size=64, epochs=160, validation_data=(X_test_scaled, y_test), verbose=1, callbacks=[ProgressCallback()])

    # Predict on the training and test sets
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Use raw continuous predictions for sma_position in range [-1, 1]
    y_train_pred_continuous = y_train_pred.flatten()
    y_test_pred_continuous = y_test_pred.flatten()

    # Calculate performance metrics using continuous predictions (e.g., MSE, MAE)
    mse = np.mean((y_test - y_test_pred_continuous) ** 2)
    mae = np.mean(np.abs(y_test - y_test_pred_continuous))

    # Print metrics
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")

    # Update progress with predictions and test data
    training_progress['test_predictions'] = y_test_pred_continuous.tolist()
    training_progress['test_actual'] = y_test.tolist()
    training_progress['status'] = 'completed'
    
    # Prepare data for CSV download: combine original data with model predictions
    # Align predictions with original datetime indices for both training and test sets
    data_with_predictions = data.copy()
    data_with_predictions['model_output'] = np.nan  # Initialize with NaN
    # Map training predictions back to original indices
    train_indices = range(len(X_train))
    for idx, pred in zip(train_indices, y_train_pred_continuous):
        if idx < len(data_with_predictions):
            data_with_predictions.loc[idx, 'model_output'] = pred
    # Map test predictions back to original indices
    test_indices = range(len(X_train), len(X_train) + len(X_test))
    for idx, pred in zip(test_indices, y_test_pred_continuous):
        if idx < len(data_with_predictions):
            data_with_predictions.loc[idx, 'model_output'] = pred
    training_progress['data_with_predictions'] = data_with_predictions.to_dict()

    # Save the model if needed
    # model.save('lstm_model.h5')

if __name__ == '__main__':
    # Start training in a background thread
    training_thread = threading.Thread(target=train_model)
    training_thread.daemon = True
    training_thread.start()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=8080, debug=False)
