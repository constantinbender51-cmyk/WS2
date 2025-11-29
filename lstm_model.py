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

# Download the CSV file from Google Drive
url = 'https://drive.google.com/uc?id=1QsfDXX4ueu4_IkPKp36EnDHnpZUR19yG'
output = 'data.csv'
gdown.download(url, output, quiet=False)

# Load the CSV data
data = pd.read_csv(output)

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

# Create sequences of 365 days for features
X = []
y = []
for i in range(365, len(close_prices)):
    X.append(close_prices[i-365:i])
    y.append(sma_positions[i])

X = np.array(X)
y = np.array(y)

# Reshape X for LSTM input: (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Standardize the features
scaler = StandardScaler()
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
scaler.fit(X_train_reshaped)
X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

# Build the LSTM model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X_train_scaled.shape[1], 1)),
    Dropout(0.5),
    LSTM(100, return_sequences=False),
    Dropout(0.5),
    Dense(50, activation='relu', kernel_regularizer=l1_l2(l1=1e-4, l2=1e-4)),
    Dense(1, activation='linear', kernel_regularizer=l1_l2(l1=1e-4, l2=1e-4))  # Linear activation for regression; adjust if sma_position is categorical
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # Adjust loss if needed for classification

# Train the model
history = model.fit(X_train_scaled, y_train, batch_size=256, epochs=20, validation_data=(X_test_scaled, y_test), verbose=1)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# If sma_position is categorical (e.g., -1, 0, 1), round predictions to nearest integer
y_pred_rounded = np.round(y_pred).astype(int)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred_rounded)
precision = precision_score(y_test, y_pred_rounded, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred_rounded, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred_rounded, average='weighted', zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred_rounded)

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Save the model if needed
# model.save('lstm_model.h5')