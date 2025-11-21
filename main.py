import asyncio
import aiohttp
import pandas as pd
import csv
import os
from datetime import datetime, timedelta
from aiohttp import web
import aiofiles
import json

class BinanceDataCollector:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.csv_filename = "binance_data.csv"
        self.data = []
        
    async def fetch_symbols(self):
        """Fetch all available trading symbols from Binance"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/exchangeInfo") as response:
                    if response.status == 200:
                        data = await response.json()
                        symbols = [symbol['symbol'] for symbol in data['symbols'] 
                                 if symbol['status'] == 'TRADING']
                        return symbols
                    else:
                        print(f"Error fetching symbols: {response.status}")
                        return []
        except Exception as e:
            print(f"Exception fetching symbols: {e}")
            return []
    
    async def fetch_klines(self, session, symbol, interval='1m', limit=50):
        """Fetch klines (candlestick) data for a specific symbol"""
        try:
            url = f"{self.base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    klines_data = await response.json()
                    return symbol, klines_data
                else:
                    print(f"Error fetching data for {symbol}: {response.status}")
                    return symbol, None
        except Exception as e:
            print(f"Exception fetching data for {symbol}: {e}")
            return symbol, None
    
    async def collect_all_data(self):
        """Collect data for all symbols"""
        print("Fetching available symbols...")
        symbols = await self.fetch_symbols()
        
        if not symbols:
            print("No symbols found!")
            return
        
        print(f"Found {len(symbols)} symbols. Fetching candle data...")
        
        # Limit to first 20 symbols to avoid too many requests
        symbols = symbols[:20]
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for symbol in symbols:
                task = asyncio.create_task(self.fetch_klines(session, symbol))
                tasks.append(task)
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.1)
            
            results = await asyncio.gather(*tasks)
            
            for symbol, klines_data in results:
                if klines_data:
                    for kline in klines_data:
                        self.data.append({
                            'symbol': symbol,
                            'open_time': kline[0],
                            'open': kline[1],
                            'high': kline[2],
                            'low': kline[3],
                            'close': kline[4],
                            'volume': kline[5],
                            'close_time': kline[6],
                            'quote_asset_volume': kline[7],
                            'number_of_trades': kline[8],
                            'taker_buy_base_asset_volume': kline[9],
                            'taker_buy_quote_asset_volume': kline[10]
                        })
        
        print(f"Collected data for {len(self.data)} candles")
    
    def save_to_csv(self):
        """Save collected data to CSV file"""
        if not self.data:
            print("No data to save!")
            return False
        
        try:
            with open(self.csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['symbol', 'open_time', 'open', 'high', 'low', 'close', 
                            'volume', 'close_time', 'quote_asset_volume', 
                            'number_of_trades', 'taker_buy_base_asset_volume', 
                            'taker_buy_quote_asset_volume']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for row in self.data:
                    writer.writerow(row)
            
            print(f"Data saved to {self.csv_filename}")
            return True
        except Exception as e:
            print(f"Error saving to CSV: {e}")
            return False
    
    def get_dataframe(self):
        """Return data as pandas DataFrame"""
        return pd.DataFrame(self.data)

async def handle_download(request):
    """Handle file download requests"""
    collector = request.app['collector']
    
    if not os.path.exists(collector.csv_filename):
        return web.Response(text="Data file not found. Please collect data first.", status=404)
    
    return web.FileResponse(collector.csv_filename, headers={
        'Content-Disposition': f'attachment; filename="{collector.csv_filename}"'
    })

async def handle_root(request):
    """Handle main page requests"""
    collector = request.app['collector']
    
    # Create HTML response
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Binance Data Collector</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
            .data-section { margin: 20px 0; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .download-btn { 
                background: #007bff; 
                color: white; 
                padding: 10px 20px; 
                text-decoration: none; 
                border-radius: 5px; 
                display: inline-block;
                margin: 10px 0;
            }
            .refresh-btn {
                background: #28a745;
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 5px;
                display: inline-block;
                margin: 10px 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Binance Data Collector</h1>
                <p>Real-time 1-minute candle data for Binance trading pairs</p>
            </div>
    """
    
    # Add download link if data exists
    if os.path.exists(collector.csv_filename):
        html_content += f"""
            <div>
                <a href="/download" class="download-btn">Download CSV File</a>
                <a href="/refresh" class="refresh-btn">Refresh Data</a>
            </div>
        """
    
    # Add data table if data exists
    if collector.data:
        df = collector.get_dataframe()
        
        # Convert timestamp to readable date
        if not df.empty and 'open_time' in df.columns:
            df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
        
        html_content += """
            <div class="data-section">
                <h2>Collected Data (First 50 rows)</h2>
                <div style="overflow-x: auto;">
        """
        
        # Convert DataFrame to HTML table (first 50 rows)
        html_content += df.head(50).to_html(classes='data-table', index=False, escape=False)
        
        html_content += """
                </div>
                <p>Total records: """ + str(len(collector.data)) + """</p>
            </div>
        """
    else:
        html_content += """
            <div>
                <p>No data collected yet. <a href="/refresh">Click here to collect data</a></p>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    return web.Response(text=html_content, content_type='text/html')

async def handle_refresh(request):
    """Handle data refresh requests"""
    collector = request.app['collector']
    
    print("Refreshing data...")
    await collector.collect_all_data()
    collector.save_to_csv()
    
    # Redirect back to main page
    return web.HTTPFound('/')

async def init_app():
    """Initialize the web application"""
    # Create data collector instance
    collector = BinanceDataCollector()
    
    # Collect initial data
    await collector.collect_all_data()
    collector.save_to_csv()
    
    # Create web application
    app = web.Application()
    app['collector'] = collector
    
    # Setup routes
    app.router.add_get('/', handle_root)
    app.router.add_get('/download', handle_download)
    app.router.add_get('/refresh', handle_refresh)
    
    return app

def main():
    """Main function to run the application"""
    print("Starting Binance Data Collector...")
    print("This will:")
    print("1. Fetch available symbols from Binance")
    print("2. Collect 50 minutes of 1-minute candle data for each symbol")
    print("3. Save data to CSV file")
    print("4. Start web server on http://localhost:8080")
    print("5. Display data and provide download link")
    
    # Run the application
    app = asyncio.run(init_app())
    
    print("\nWeb server started! Visit http://localhost:8080")
    print("Press Ctrl+C to stop the server")
    
    web.run_app(app, host='localhost', port=8080)

if __name__ == "__main__":
    main()
