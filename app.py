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
            padding: 20px;
            background-color: white;
            font-family: Arial, sans-serif;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        img {
            display: block;
            margin: 20px auto;
            max-width: 100%;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Random Forest Predicting a Flat Line</h1>
        <img src="data:image/png;base64,{{ img_data }}" alt="Line prediction">
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    # Generate flat line data
    n_points = 100
    X_train = np.arange(n_points).reshape(-1, 1)
    y_train = np.ones(n_points) * 5  # Flat line at y=5
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Predict continuation
    X_predict = np.arange(n_points, n_points * 2).reshape(-1, 1)
    y_predict = rf.predict(X_predict)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot original data (flat line)
    ax.plot(X_train, y_train, 'b-', linewidth=2, label='Original Data')
    
    # Plot vertical separator
    ax.axvline(x=n_points, color='red', linestyle='--', linewidth=2, label='Separator')
    
    # Plot prediction
    ax.plot(X_predict, y_predict, 'g-', linewidth=2, label='Random Forest Prediction')
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Random Forest: Because Even AI Knows a Flat Line When It Sees One', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 10)
    
    # Convert plot to base64 image
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', facecolor='white')
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.read()).decode()
    plt.close()
    
    return render_template_string(HTML_TEMPLATE, img_data=img_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)