import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from io import BytesIO
from flask import Flask, Response

app = Flask(__name__)

@app.route('/')
def plot():
    # Straight line: y = 50 from x=0 to x=10
    x_train = np.linspace(0, 10, 100).reshape(-1, 1)
    y_train = np.full(100, 50.0)

    # Train RF
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(x_train, y_train)

    # Predict continuation: x=10 to x=20
    x_pred = np.linspace(10, 20, 100).reshape(-1, 1)
    y_pred = model.predict(x_pred)

    # Combine
    x_combined = np.concatenate([x_train.flatten(), x_pred.flatten()])
    y_combined = np.concatenate([y_train, y_pred])

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')
    
    # Main line (black)
    ax.plot(x_combined, y_combined, color='black', linewidth=2)
    
    # Vertical line at prediction start (x=10)
    ax.axvline(x=10, color='black', linewidth=1.5, linestyle='-')

    # Kill all elements
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