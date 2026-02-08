import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from flask import Flask, Response

app = Flask(__name__)

@app.route('/')
def plot():
    # Create a clean white canvas
    plt.figure(figsize=(12, 6), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # Generate a smooth line (this is your "SMA line" - simplified for demo)
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * 10 + 50  # This is your "SMA line"
    
    # Plot ONLY the line (no grid, no labels, no extra shit)
    plt.plot(x, y, color='blue', linewidth=2.5)
    
    # Remove all axes and borders
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Save as PNG
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()
    return Response(buf.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)