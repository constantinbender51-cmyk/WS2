import pandas as pd
import gdown
import os
from flask import Flask, send_file, render_template_string
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import MinMaxScaler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Google Drive file ID extracted from the URL
file_id = '1kDCl_29nXyW1mLNUAS-nsJe0O2pOuO6o'
url = f'https://drive.google.com/uc?id={file_id}'
csv_filename = 'data.csv'
processed_filename = 'processed_data.csv'
plot_filename = 'data_plot.png'

# Global variables to store precomputed data
processing_details = None
plot_image_base64 = None

# Define the processing function first



# HTML template for the web page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Processed CSV Download</title>
</head>
<body>
    <h1>Processed CSV Data</h1>
    <p>The CSV has been downloaded, processed, and resampled to 5-minute intervals with range and volume metrics computed.</p>
    <p><a href="/download">Download Processed CSV</a></p>
    <h2>Data Plot:</h2>
    <img src="data:image/png;base64,{{ plot_image }}" alt="Data Plot" style="max-width: 100%; height: auto;">
    <p><a href="/plot">Download Plot Image</a></p>
    <h2>Processing Details:</h2>
    <pre>{{ details }}</pre>
</body>
</html>
"""


def download_and_process_csv():
    """Download CSV from Google Drive and process it."""
    try:
        # Download the file
        logger.info(f"Downloading CSV from {url}...")
        gdown.download(url, csv_filename, quiet=False)
        if not os.path.exists(csv_filename):
            raise FileNotFoundError(f"Download failed: {csv_filename} not found.")
        logger.info(f"Downloaded to {csv_filename}")
        
        # Load CSV
        df = pd.read_csv(csv_filename)
        logger.info(f"Loaded CSV with shape: {df.shape}")
        
        # Get column names
        columns = df.columns.tolist()
        logger.info(f"Columns in CSV: {columns}")
        
        # Identify datetime column (assumed to be the first column)
        datetime_col = columns[0] if len(columns) > 0 else None
        if datetime_col is None:
            raise ValueError("CSV has no columns.")
        logger.info(f"Using datetime column: {datetime_col}")
        
        # Convert datetime column to datetime
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        df.set_index(datetime_col, inplace=True)
        
        # Identify OHLCV columns (case-insensitive search)
        expected_cols = ['open', 'high', 'low', 'close', 'volume']
        col_map = {}
        for exp in expected_cols:
            matches = [col for col in columns if exp.lower() in col.lower()]
            if matches:
                col_map[exp] = matches[0]
            else:
                raise ValueError(f"Column '{exp}' not found in CSV. Available columns: {columns}")
        logger.info(f"Mapped columns: {col_map}")
        
        # Ensure required columns are present
        required = ['open', 'high', 'low', 'close', 'volume']
        for req in required:
            if req not in col_map:
                raise ValueError(f"Required column '{req}' not found.")
        
        # Resample to 5-minute OHLCV data
        resampled = df.resample('5T').agg({
            col_map['open']: 'first',
            col_map['high']: 'max',
            col_map['low']: 'min',
            col_map['close']: 'last',
            col_map['volume']: 'sum'
        })
        
        # Rename columns to standard names
        resampled.rename(columns={
            col_map['open']: 'open',
            col_map['high']: 'high',
            col_map['low']: 'low',
            col_map['close']: 'close',
            col_map['volume']: 'volume'
        }, inplace=True)
        
        # Compute range (high - low)
        resampled['range'] = resampled['high'] - resampled['low']
        
        # Compute range divided by volume, handling potential division by zero
        # Replace 0 volume with pd.NA to avoid division by zero, and ensure range is not NA
        # If range is NA or volume is NA (or 0), the result should be NA
        resampled['range_to_volume'] = resampled['range'] / resampled['volume']
        resampled.loc[(resampled['range'].isna()) | (resampled['volume'] == 0) | (resampled['volume'].isna()), 'range_to_volume'] = pd.NA

        # Calculate the maximum range_to_volume over the last 4 days (1152 intervals)
        # 4 days * 24 hours/day * 60 minutes/hour = 5760 minutes
        # 5760 minutes / 5 minutes per interval = 1152 intervals
        window_size = 1152
        if len(resampled) >= window_size:
            resampled['max_range_vol_4d'] = resampled['range_to_volume'].rolling(window=window_size, min_periods=1).max()
        else:
            resampled['max_range_vol_4d'] = pd.NA # Handle cases with insufficient data
        
        # Reset index to make datetime a column
        resampled.reset_index(inplace=True)
        
        # Save processed CSV
        resampled.to_csv(processed_filename, index=False)
        logger.info(f"Processed data saved to {processed_filename} with shape: {resampled.shape}")
        
        # Prepare processing details for display
        details = f"""
Original CSV shape: {df.shape}
Processed CSV shape: {resampled.shape}
Datetime column: {datetime_col}
Resample frequency: 5 minutes
Columns in processed data: {list(resampled.columns)}
First few rows:
{resampled.head().to_string()}
"""
        
        # Filter out NaNs for plotting to avoid errors
        plot_data_close = resampled.dropna(subset=['close'])
        plot_data_metrics = resampled.dropna(subset=['range', 'range_to_volume'])
        
        # Filter out NaNs for plotting to avoid errors
        plot_data_close = resampled.dropna(subset=['close'])
        plot_data_metrics = resampled.dropna(subset=['range', 'range_to_volume', 'max_range_vol_4d'])

        # Apply Min-Max scaling to 'range_to_volume' and 'max_range_vol_4d' for the secondary axis
        scaler = MinMaxScaler()
        metrics_to_scale = ['range_to_volume', 'max_range_vol_4d']
        
        # Ensure we only attempt to scale columns that actually exist in plot_data_metrics
        existing_metrics_to_scale = [col for col in metrics_to_scale if col in plot_data_metrics.columns]
        
        if existing_metrics_to_scale:
            # Perform scaling
            scaled_values = scaler.fit_transform(plot_data_metrics[existing_metrics_to_scale])
            
            # Create new columns for scaled data and add them to plot_data_metrics
            scaled_df = pd.DataFrame(scaled_values, columns=[f'{col}_scaled' for col in existing_metrics_to_scale], index=plot_data_metrics.index)
            plot_data_metrics = pd.concat([plot_data_metrics, scaled_df], axis=1)
        
        # Create matplotlib plot with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Plot 'close' price on the primary y-axis (ax1)
        ax1.plot(plot_data_close.index, plot_data_close['close'], label='Close Price', color='blue', linewidth=1)
        ax1.set_xlabel('Date/Time')
        ax1.set_ylabel('Close Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Create a secondary y-axis for scaled metrics ('range_to_volume' and SMAs)
        ax2 = ax1.twinx()
        
        # Plot scaled 'range_to_volume' on the secondary y-axis
        if 'range_to_volume_scaled' in plot_data_metrics.columns:
            ax2.plot(plot_data_metrics.index, plot_data_metrics['range_to_volume_scaled'], label='Scaled Range/Volume (0-1)', color='orange', linewidth=1)
        
        # Plot scaled 'range_to_volume' and 'max_range_vol_4d' on the secondary y-axis
        if 'range_to_volume_scaled' in plot_data_metrics.columns:
            ax2.plot(plot_data_metrics.index, plot_data_metrics['range_to_volume_scaled'], label='Scaled Range/Volume (0-1)', color='orange', linewidth=1)
        if 'max_range_vol_4d_scaled' in plot_data_metrics.columns:
            ax2.plot(plot_data_metrics.index, plot_data_metrics['max_range_vol_4d_scaled'], label='Max Range/Volume (4d, 0-1)', color='green', linewidth=1, linestyle='-.')

        ax2.set_ylabel('Scaled Value (0-1)', color='purple') # Using purple for this axis label
        ax2.tick_params(axis='y', labelcolor='purple')
        ax2.set_ylim(0, 1) # Ensure Y axis for scaled data spans from 0 to 1

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

        plt.title('Close Price and Range/Volume Metrics')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout() # Adjust layout to prevent overlap
        
        # Save plot to a bytes buffer and encode as base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        plot_image = base64.b64encode(buf.read()).decode('utf-8')
        
        # Save plot to file for download
        with open(plot_filename, 'wb') as f:
            f.write(base64.b64decode(plot_image))
        
        return details, plot_image
    
    except Exception as e:
        logger.error(f"Error in processing: {e}")
        raise


@app.route('/')
def index():
    """Main page with download link, SMA plot, and processing details."""
    if processing_details is None or plot_image_base64 is None:
        return "<h1>Error</h1><p>Data processing failed at startup. Check server logs.</p>", 500
    return render_template_string(HTML_TEMPLATE, details=processing_details, plot_image=plot_image_base64)


@app.route('/download')
def download():
    """Endpoint to download the processed CSV."""
    if not os.path.exists(processed_filename):
        return "Processed file not found. Please visit the main page first.", 404
    return send_file(processed_filename, as_attachment=True, download_name='processed_5min_data.csv')

@app.route('/plot')
def plot_download():
    """Endpoint to download the plot image."""
    if not os.path.exists(plot_filename):
        return "Plot file not found. Please visit the main page first.", 404
    return send_file(plot_filename, as_attachment=True, download_name='data_plot.png')


if __name__ == '__main__':
    # Perform computation at startup
    try:
        logger.info("Starting data processing at startup...")
        processing_details, plot_image_base64 = download_and_process_csv()
        logger.info("Data processing completed successfully at startup.")
    except Exception as e:
        logger.error(f"Startup processing failed: {e}")
        processing_details = f"Startup processing failed: {e}"
        plot_image_base64 = None
    
    # Run Flask app on port 8080
    logger.info("Starting web server on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False)
