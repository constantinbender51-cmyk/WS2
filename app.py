import pandas as pd
import requests
from flask import Flask, send_file, render_template_string
import io
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import base64

app = Flask(__name__)

# Function to fetch OHLCV data from Binance
def fetch_binance_data(symbol='BTCUSDT', start_date='2018-01-01', end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Convert dates to milliseconds for Binance API
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': '1d',
        'startTime': start_ts,
        'endTime': end_ts,
        'limit': 1000
    }
    
    all_data = []
    while start_ts < end_ts:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if not data:
                break
            all_data.extend(data)
            # Update start_ts to the next day after the last entry
            start_ts = data[-1][0] + 86400000  # Add 1 day in milliseconds
            params['startTime'] = start_ts
        else:
            print(f"Error fetching data: {response.status_code}")
            break
    
    # Convert to DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    df = pd.DataFrame(all_data, columns=columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]  # Keep relevant columns
    df.set_index('timestamp', inplace=True)
    return df

# Function to calculate SMA position
def calculate_sma_position(df):
    df['sma_120'] = df['close'].rolling(window=120).mean()
    df['sma_365'] = df['close'].rolling(window=365).mean()
    
    # Initialize sma_position column
    df['sma_position'] = 0
    
    # Conditions for sma_position
    # 1 if 365 SMA below current price and not (below 365 SMA and above 120 SMA)
    condition1 = (df['sma_365'] < df['close']) & ~((df['sma_365'] < df['close']) & (df['close'] > df['sma_120']))
    # -1 if 365 SMA above current price and not (above 365 SMA and below 120 SMA)
    condition2 = (df['sma_365'] > df['close']) & ~((df['sma_365'] > df['close']) & (df['close'] < df['sma_120']))
    
    df.loc[condition1, 'sma_position'] = 1
    df.loc[condition2, 'sma_position'] = -1
    
    # Note: The conditions above simplify the logic based on the description; sma_position remains 0 in other cases.
    return df

# Generate the data and CSV
def generate_data():
    df = fetch_binance_data()
    df = calculate_sma_position(df)
    # Reset index to include timestamp as a column
    df.reset_index(inplace=True)
    return df

# Generate plot
def generate_plot():
    df = generate_data()
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['close'], label='Close Price', color='blue')
    plt.title('Binance OHLCV Data - Close Price and SMA Position')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot to a bytes buffer
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    # Encode image to base64 for embedding in HTML
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url

@app.route('/')
def index():
    return '<h1>Binance OHLCV Data with SMA Position</h1><p><a href="/download">Download CSV</a></p><p><a href="/plot">View Plot</a></p>'

@app.route('/download')
def download_csv():
    df = generate_data()
    # Exclude SMA columns from CSV
    df_csv = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'sma_position']]
    output = io.BytesIO()
    df_csv.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, as_attachment=True, download_name='binance_data_sma.csv', mimetype='text/csv')

@app.route('/plot')
def plot_data():
    plot_url = generate_plot()
    html_content = f'''
    <h1>Binance OHLCV Data Plot</h1>
    <img src="data:image/png;base64,{plot_url}" alt="OHLCV Plot">
    <p><a href="/">Back to Home</a></p>
    '''
    return render_template_string(html_content)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)