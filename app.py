import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from flask import Flask, send_file
import io

app = Flask(__name__)

@app.route('/')
def serve_line():
    # 1. Generate complicated noisy data (Non-linear)
    x = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.sin(x).ravel() + np.random.normal(0, 0.2, x.shape[0]) + (x.ravel() * 0.1)

    # 2. ML Prediction (Neural Network)
    # Preprocessing
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_scaled = scaler_x.fit_transform(x)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # Train MLP
    model = MLPRegressor(hidden_layer_sizes=(100, 100), activation='tanh', 
                         solver='lbfgs', max_iter=2000, random_state=42)
    model.fit(x_scaled, y_scaled)

    # Predict continuation
    future_steps = 30
    x_future = np.linspace(10, 13, future_steps).reshape(-1, 1)
    x_future_scaled = scaler_x.transform(x_future)
    y_future_scaled = model.predict(x_future_scaled)
    y_future = scaler_y.inverse_transform(y_future_scaled.reshape(-1, 1)).ravel()

    # 3. Plotting
    plt.figure(figsize=(12, 6), frameon=False)
    
    # Original Data
    plt.plot(x, y, color='black', linewidth=2)
    
    # Vertical Line at end of known data
    plt.axvline(x=x[-1], color='red', linewidth=2, linestyle='--')
    
    # Predicted Continuation
    plt.plot(x_future, y_future, color='blue', linewidth=2, linestyle='-')

    # Remove all axis/chrome
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])

    # Serve
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(port=8080)
