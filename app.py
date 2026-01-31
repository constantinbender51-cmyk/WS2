import requests
import json

def fetch_futures_orderbook(symbol: str):
    # Kraken Futures v3 API Endpoint
    url = "https://futures.kraken.com/derivatives/api/v3/orderbook"
    
    # Parameters
    params = {
        'symbol': symbol.upper()  # specific symbol required (e.g., PF_BTCUSD)
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Kraken Futures API returns a 'result' key indicating success
        if data.get('result') == 'success':
            ob = data.get('orderBook', {})
            bids = ob.get('bids', [])
            asks = ob.get('asks', [])

            print(f"--- Order Book for {symbol.upper()} ---")
            
            # Display best 5 asks (lowest prices first)
            # Asks are sorted ascending by price
            print("\nTop 5 Asks (Price | Size):")
            for ask in asks[:5]:
                print(f"{ask[0]:<10} | {ask[1]}")

            # Display best 5 bids (highest prices first)
            # Bids are sorted descending by price
            print("\nTop 5 Bids (Price | Size):")
            for bid in bids[:5]:
                print(f"{bid[0]:<10} | {bid[1]}")
                
            print(f"\nTotal Asks: {len(asks)}")
            print(f"Total Bids: {len(bids)}")
            
        else:
            print(f"API Error: {data}")

    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed: {e}")

if __name__ == "__main__":
    fetch_futures_orderbook("PF_XBTUSD")
