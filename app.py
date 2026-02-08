import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from io import BytesIO
from flask import Flask, Response

app = Flask(__name__)

@app.route('/')
def plot():
    # Generate -10° straight line
    angle_rad = np.deg2rad(10)
    slope = -np.tan(angle_rad)  # Negative for downward slope
    y = 50 + slope * np.arange(200)  # 200 points: 0→199
    
    # Features: [y_{t-2}, y_{t-1}] → predict y_t
    X, target = [], []
    for i in range(2, 100):  # Train on first 100 points (indices 2-99)
        X.append([y[i-2], y[i-1]])
        target.append(y[i])
    
    # Train RF on the slope pattern (delta between last 2 points)
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=0)
    model.fit(X, target)
    
    # Roll forward from point 100 to 199 using ONLY predicted values
    y_pred = y[:100].tolist()  # Start with real data (first 100 points)
    for i in range(100, 200):
        next_val = model.predict([[y_pred[-2], y_pred[-1]]])[0]
        y_pred.append(next_val)
    
    # Plot: one black line, vertical marker at prediction start (point 100)
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')
    ax.plot(range(200), y_pred, color='black', linewidth=2)
    ax.axvline(x=100, color='black', linewidth=1.5)
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()
    return Response(buf.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)