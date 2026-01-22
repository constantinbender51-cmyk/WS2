import requests
import matplotlib.pyplot as plt
import http.server
import socketserver
import io
import os

# 1. Fetch Data
URL = "https://workspace-production-9fae.up.railway.app/history"
response = requests.get(URL)
data = response.json()

# 2. Process Data
# Extract PnL and calculate cumulative returns for plotting
pnls = [item['pnl'] for item in data]
cumulative_pnl = [sum(pnls[:i+1]) for i in range(len(pnls))]
timestamps = [item['time'].split('T')[1][:5] for item in data] # Shorten time for X-axis

# 3. Generate Plot
plt.figure(figsize=(10, 6))
plt.plot(timestamps, cumulative_pnl, marker='o', linestyle='-', color='b')
plt.title('Cumulative PnL Over Time')
plt.xlabel('Time')
plt.ylabel('PnL')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Save plot to current directory
plot_filename = "pnl_plot.png"
plt.savefig(plot_filename)
plt.close()

# 4. Serve on Port 8080
PORT = 8080

class PlotHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(f'<html><body><h1>PnL History Plot</h1><img src="{plot_filename}"></body></html>'.encode())
        else:
            super().do_GET()

print(f"Serving PnL plot at http://localhost:{PORT}")
with socketserver.TCPServer(("", PORT), PlotHandler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
