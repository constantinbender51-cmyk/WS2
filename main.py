import os
import pandas as pd
import gdown
from flask import Flask, send_file, render_template_string
import threading
import time

# Google Drive file ID extracted from the URL
FILE_ID = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
DOWNLOAD_URL = f'https://drive.google.com/uc?id={FILE_ID}'
INPUT_FILENAME = 'ohlcv_1min.csv'
OUTPUT_FILENAME = 'ohlcv_all_timeframes.csv'

# Timeframes for resampling in minutes (5 minutes is the baseline, so start from 10 minutes)
TIMEFRAMES = {
    '10min': '10T',
    '30min': '30T',
    'hourly': '1H',
    '4hourly': '4H',
    'daily': '1D',
    'weekly': '1W'
}

def download_file():
    """Download the CSV file from Google Drive using gdown"""
    print(f"Downloading file from {DOWNLOAD_URL}...")
    try:
        gdown.download(DOWNLOAD_URL, INPUT_FILENAME, quiet=False)
        if os.path.exists(INPUT_FILENAME):
            print(f"File downloaded successfully: {INPUT_FILENAME}")
            return True
        else:
            print("Download failed: File not found after download")
            return False
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def process_data():
    """Load the 1-minute OHLCV data and resample to multiple timeframes"""
    print("Loading and processing data...")
    
    # Load the CSV file
    try:
        df = pd.read_csv(INPUT_FILENAME)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None
    
    # Check required columns, handle 'datetime' as an alternative to 'timestamp'
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    timestamp_col = None
    if 'timestamp' in df.columns:
        timestamp_col = 'timestamp'
    elif 'datetime' in df.columns:
        timestamp_col = 'datetime'
    else:
        print("Missing timestamp column: expected 'timestamp' or 'datetime'")
        return None
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return None
    
    # Convert timestamp to datetime and set as index
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.set_index(timestamp_col, inplace=True)
    df.index.name = 'timestamp'  # Standardize index name for consistency
    
    # Ensure data is sorted by timestamp
    df.sort_index(inplace=True)
    
    # Note: 1-minute data is loaded but will be resampled to 5 minutes as baseline
    
    # Dictionary to store resampled dataframes
    resampled_dfs = {}
    
    # Resample for each timeframe
    for tf_name, tf_code in TIMEFRAMES.items():
        print(f"Resampling to {tf_name} timeframe...")
        
        # Resample OHLCV data
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        resampled = df.resample(tf_code).apply(ohlc_dict)
        
        # Flatten multi-level columns
        resampled.columns = [f'{col}_{tf_name}' for col in resampled.columns]
        
        # Reset index to have timestamp as a column
        resampled.reset_index(inplace=True)
        
        resampled_dfs[tf_name] = resampled
    
    # Merge all dataframes on timestamp
    print("Merging all timeframes...")
    
    # Start with the 5-minute baseline data (resample 1-minute to 5 minutes first)
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    df_5min = df.resample('5T').apply(ohlc_dict)
    df_5min.columns = [f'{col}_5min' for col in df_5min.columns]
    df_5min.reset_index(inplace=True)
    
    # Merge all dataframes
    merged_df = df_5min
    for tf_name, tf_df in resampled_dfs.items():
        merged_df = pd.merge(merged_df, tf_df, on='timestamp', how='outer')
    
    # Sort by timestamp
    merged_df.sort_values('timestamp', inplace=True)
    
    # Save to CSV
    print(f"Saving combined data to {OUTPUT_FILENAME}...")
    merged_df.to_csv(OUTPUT_FILENAME, index=False)
    
    print(f"Data processing complete. File saved: {OUTPUT_FILENAME}")
    print(f"Total rows: {len(merged_df)}")
    print(f"Total columns: {len(merged_df.columns)}")
    print(f"Note: Original data had {len(df)} entries, which may affect resampling for longer timeframes.")
    
    return merged_df

# Flask web server
def start_web_server():
    """Start a Flask web server to serve the download link"""
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        """Display download link for the CSV file"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>OHLCV Data Download</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #333;
                }
                .download-link {
                    display: inline-block;
                    background-color: #4CAF50;
                    color: white;
                    padding: 15px 30px;
                    text-decoration: none;
                    border-radius: 5px;
                    font-size: 18px;
                    margin: 20px 0;
                }
                .download-link:hover {
                    background-color: #45a049;
                }
                .info {
                    background-color: #f9f9f9;
                    padding: 15px;
                    border-left: 4px solid #4CAF50;
                    margin: 20px 0;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>OHLCV Data Download</h1>
                <div class="info">
                    <p>This file contains OHLCV (Open, High, Low, Close, Volume) data resampled to multiple timeframes:</p>
                    <ul>
                        <li>5 minutes (baseline)</li>
                        <li>10 minutes</li>
                        <li>30 minutes</li>
                        <li>Hourly</li>
                        <li>4-hourly</li>
                        <li>Daily</li>
                        <li>Weekly</li>
                    </ul>
                </div>
                <p>Click the button below to download the CSV file:</p>
                <a class="download-link" href="/download">Download ohlcv_all_timeframes.csv</a>
                <p>File size: {{ file_size }} MB</p>
                <p>Total rows: {{ row_count }}</p>
                <p>Total columns: {{ column_count }}</p>
            </div>
        </body>
        </html>
        """
        
        # Get file info
        file_size = 0
        row_count = 0
        column_count = 0
        
        if os.path.exists(OUTPUT_FILENAME):
            file_size = os.path.getsize(OUTPUT_FILENAME) / (1024 * 1024)  # Convert to MB
            file_size = round(file_size, 2)
            
            # Try to get row and column count
            try:
                df = pd.read_csv(OUTPUT_FILENAME)
                row_count = len(df)
                column_count = len(df.columns)
            except:
                pass
        
        return render_template_string(html_template, 
                                     file_size=file_size,
                                     row_count=row_count,
                                     column_count=column_count)
    
    @app.route('/download')
    def download_file():
        """Serve the CSV file for download"""
        if os.path.exists(OUTPUT_FILENAME):
            return send_file(OUTPUT_FILENAME, 
                           as_attachment=True, 
                           download_name=OUTPUT_FILENAME)
        else:
            return "File not found. Please wait for data processing to complete.", 404
    
    print("Starting web server on http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)

def main():
    """Main function to orchestrate the process"""
    print("Starting OHLCV data processing pipeline...")
    
    # Step 1: Download the file
    if not download_file():
        print("Failed to download file. Exiting.")
        return
    
    # Step 2: Process the data
    processed_data = process_data()
    if processed_data is None:
        print("Failed to process data. Exiting.")
        return
    
    # Step 3: Start the web server
    print("\nData processing complete. Starting web server...")
    start_web_server()

if __name__ == '__main__':
    main()