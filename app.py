import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from io import BytesIO
from flask import Flask, Response

app = Flask(__name__)

@app.route('/')
def plot():
    # Training data: straight downward line (x=0 to x=10)
    x_train = np.linspace(0, 10, 100).reshape(-1, 1)
    y_train = 60 - 1.0 * x_train.flatten()  # Drops 1 unit per x
    
    # Train RF on (x, y) pairs
    model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=0)
    model.fit(x_train, y_train)
    
    # Predict ALL future points at once (x=10 to x=20) — NO recursion
    x_pred = np.linspace(10, 20, 100).reshape(-1, 1)
    y_pred = model.predict(x_pred)
    
    # Combine
    x_all = np.concatenate([x_train.flatten(), x_pred.flatten()])
    y_all = np.concatenate([y_train, y_pred])
    
    # Plot: pure white, black lines only
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')
    ax.plot(x_all, y_all, color='black', linewidth=2)          # Main line
    ax.axvline(x=10, color='black', linewidth=1.5)             # Prediction start
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()
    return Response(buf.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)