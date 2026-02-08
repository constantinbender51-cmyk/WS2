import matplotlib.pyplot as plt
from io import BytesIO
from flask import Flask, Response

app = Flask(__name__)

@app.route('/')
def plot():
    # Create pure white canvas
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    
    # Plot ONLY a single black line
    ax.plot([0.1, 0.9], [0.5, 0.5], color='black', linewidth=2)
    
    # KILL ALL ELEMENTS
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Save with ZERO padding
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='white')
    buf.seek(0)
    plt.close()
    return Response(buf.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)