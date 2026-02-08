import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from io import BytesIO
from flask import Flask, Response

app = Flask(__name__)

@app.route('/')
def plot():
    # Create -10° line (training data: x=0 to x=10)
    angle_rad = np.deg2rad(10)
    slope = np.tan(angle_rad)
    x_full = np.linspace(0, 20, 200)
    y_full = 50 - (x_full * slope)
    
    # Build features: [y_{t-2}, y_{t-1}] → predict y_t
    # This lets RF SEE the slope (difference between last 2 points)
    window = 2
    X, y = [], []
    for i in range(window, 100):  # Train only on first 10 units (x=0→10)
        X.append(y_full[i-window:i])  # Last 2 y-values
        y.append(y_full[i])           # Current y-value
    
    X, y = np.array(X), np.array(y)
    
    # Train RF on slope context
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X, y)
    
    # Roll forward prediction step-by-step (x=10→20)
    y_pred = list(y_full[:100])  # Start with real data
    for i in range(100, 200):
        last_points = np.array(y_pred[-window:]).reshape(1, -1)
        next_y = model.predict(last_points)[0]
        y_pred.append(next_y)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')
    
    # Real data (x=0→10) + prediction (x=10→20)
    ax.plot(x_full, y_pred, color='black', linewidth=2)
    
    # Vertical line at prediction start
    ax.axvline(x=10, color='black', linewidth=1.5)
    
    # Kill everything else
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='white', edgecolor='none',
                bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()
    return Response(buf.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)