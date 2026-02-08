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
    
    # Train on DOUBLE the range (0-200) so it sees the values we'll predict
    X_full = np.arange(n_points * 2).reshape(-1, 1)
    y_full = 5 + 0.05 * X_full.flatten()
    
    # Train Random Forest on the FULL range
    rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
    rf.fit(X_full, y_full)
    
    # Now split for display purposes
    X_raw = X_full[:n_points]
    y_train = y_full[:n_points]
    
    X_raw_predict = X_full[n_points:]
    y_predict = rf.predict(X_raw_predict)
    
    # Create plot with no decorations
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot original data (increasing line)
    ax.plot(X_raw.flatten(), y_train, 'b-', linewidth=2)
    
    # Plot vertical separator
    ax.axvline(x=n_points, color='red', linestyle='-', linewidth=2)
    
    # Plot prediction
    ax.plot(X_raw_predict.flatten(), y_predict, 'g-', linewidth=2)
    
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
