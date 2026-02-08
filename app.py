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
    # 1. Generate Data (Straight line input as requested, system capable of complex noise)
    x = np.linspace(0, 10, 100).reshape(-1, 1)
    y = (1.5 * x).ravel() + np.random.normal(0, 0.1, x.shape[0])

    # 2. ML Prediction (MLPRegressor)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_scaled = scaler_x.fit_transform(x)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    model = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', 
                         solver='adam', max_iter=5000, random_state=42)
    model.fit(x_scaled, y_scaled)

    # Predict continuation
    x_future = np.linspace(10, 13, 30).reshape(-1, 1)
    x_future_scaled = scaler_x.transform(x_future)
    y_future_scaled = model.predict(x_future_scaled)
    y_future = scaler_y.inverse_transform(y_future_scaled.reshape(-1, 1)).ravel()

    # 3. Plotting
    plt.figure(figsize=(12, 6), frameon=False)
    
    # Data line
    plt.plot(x, y, color='black', linewidth=2)
    
    # Vertical divider
    plt.axvline(x=x[-1], color='red', linewidth=2, linestyle='--')
    
    # Prediction line
    plt.plot(x_future, y_future, color='blue', linewidth=2, linestyle='-')

    # Formatting
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
