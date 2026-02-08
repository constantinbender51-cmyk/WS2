from flask import Flask, render_template_string
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Random Forest Line Prediction</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            background-color: white;
        }
        img {
            display: block;
            width: 100%;
            height: 100vh;
            object-fit: contain;
        }
    </style>
</head>
<body>
    <img src="data:image/png;base64,{{ img_data }}" alt="Line prediction">
</body>
</html>
'''

@app.route('/')
def index():
    # Generate slightly increasing line data
    n_points = 100
    X_raw = np.arange(n_points)
    y_train = 5 + 0.02 * X_raw  # Slight upward slope
    
    # Create features using LAGGED values (previous y value and position)
    # This way the RF learns: "next value = previous value + slope"
    X_train_features = []
    y_train_targets = []
    
    for i in range(1, n_points):
        X_train_features.append([y_train[i-1], X_raw[i]])  # [previous_y, current_x]
        y_train_targets.append(y_train[i])
    
    X_train_features = np.array(X_train_features)
    y_train_targets = np.array(y_train_targets)
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_features, y_train_targets)
    
    # Predict continuation - iteratively using previous predictions
    X_raw_predict = np.arange(n_points, n_points * 2)
    y_predict = []
    last_y = y_train[-1]
    
    for x in X_raw_predict:
        next_y = rf.predict([[last_y, x]])[0]
        y_predict.append(next_y)
        last_y = next_y
    
    y_predict = np.array(y_predict)
    
    # Create plot with no decorations
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot original data (increasing line)
    ax.plot(X_raw, y_train, 'b-', linewidth=2)
    
    # Plot vertical separator
    ax.axvline(x=n_points, color='red', linestyle='-', linewidth=2)
    
    # Plot prediction
    ax.plot(X_raw_predict, y_predict, 'g-', linewidth=2)
    
    # Remove all decorations
    ax.set_xlim(0, n_points * 2)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Convert plot to base64 image
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', facecolor='white')
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.read()).decode()
    plt.close()
    
    return render_template_string(HTML_TEMPLATE, img_data=img_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
