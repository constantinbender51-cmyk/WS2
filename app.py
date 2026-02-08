import io
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server integration
import matplotlib.pyplot as plt
from flask import Flask, Response
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

# 1. Data Generation & Model Training
# Non-linear domain (Sine wave)
X_train = np.sort(5 * np.random.rand(80, 1), axis=0)
y_train = np.sin(X_train).ravel()
y_train[::5] += 3 * (0.5 - np.random.rand(16)) # Add noise

# Model: SVR with RBF kernel to capture non-linearity
svr_rbf = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1))
svr_rbf.fit(X_train, y_train)

# 2. Sampling / Prediction
# Continuation of non-linear line
X_test_nonlinear = np.arange(0.0, 7.0, 0.01)[:, np.newaxis]
y_pred_nonlinear = svr_rbf.predict(X_test_nonlinear)

# Straight declining line domain (Out of distribution test)
X_test_linear = np.linspace(5, 10, 100)[:, np.newaxis]
# The "ground truth" for the declining line to compare against model's inference
y_true_linear = -1 * (X_test_linear - 5) - 1 
y_pred_linear = svr_rbf.predict(X_test_linear)

@app.route('/')
def plot_png():
    # 3. Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Training Data
    ax.scatter(X_train, y_train, color='darkorange', label='Training Data (Non-linear)', s=10, zorder=3)
    
    # Model Prediction on Non-linear continuation
    ax.plot(X_test_nonlinear, y_pred_nonlinear, color='navy', lw=2, label='RBF Model Prediction (Interpolation/Extrapolation)')
    
    # Linear Declining Line (Target vs Model Behavior)
    ax.plot(X_test_linear, y_true_linear, color='green', linestyle='--', label='Target Declining Line (Ground Truth)')
    ax.plot(X_test_linear, y_pred_linear, color='red', linestyle='-', label='Model Response to Declining Input')

    ax.set_title('SVR Response to Non-Linear Training vs Linear Declining Input')
    ax.set_xlabel('Input Feature (X)')
    ax.set_ylabel('Target (y)')
    ax.legend()
    ax.grid(True)

    # Buffer image to memory
    output = io.BytesIO()
    fig.savefig(output, format='png')
    plt.close(fig)
    output.seek(0)
    
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    # Binds strictly to 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
