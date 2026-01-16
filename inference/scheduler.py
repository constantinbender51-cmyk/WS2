import time
import subprocess
import sys
from datetime import datetime, timedelta

# List of tickers to process (matching your training script)
TICKERS = ["BTC", "ETH", "SOL"]

# Name of your existing inference script
INFERENCE_SCRIPT = "inference.py"

def get_seconds_until_next_interval(interval_minutes=15):
    """
    Calculates the number of seconds until the next aligned interval
    (e.g., :00, :15, :30, :45).
    """
    now = datetime.now()
    
    # Calculate how many minutes to add to reach the next interval
    minutes_to_add = interval_minutes - (now.minute % interval_minutes)
    
    # Create the target time
    target_time = now + timedelta(minutes=minutes_to_add)
    
    # Zero out seconds and microseconds to ensure we hit exactly XX:XX:00
    target_time = target_time.replace(second=0, microsecond=0)
    
    seconds_remaining = (target_time - now).total_seconds()
    
    # If seconds_remaining is negative (rare edge case), jump to next interval
    if seconds_remaining < 0:
        seconds_remaining += (interval_minutes * 60)
        
    return seconds_remaining

def run_batch_inference():
    """
    Runs the inference script for every ticker in the list.
    """
    print(f"\n{'='*40}")
    print(f"⏰ Triggering Batch Inference: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*40}")

    for ticker in TICKERS:
        try:
            print(f"running {ticker}...")
            # Calls: python inference.py <TICKER>
            subprocess.run(
                [sys.executable, INFERENCE_SCRIPT, ticker], 
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running inference for {ticker}: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

def main():
    print(f"--- 🗓️ Scheduler Started ---")
    print(f"Tracking tickers: {TICKERS}")
    print("Waiting for the next 15-minute mark (00, 15, 30, 45)...")

    while True:
        # 1. Calculate sleep time
        sleep_seconds = get_seconds_until_next_interval(15)
        
        # 2. Sleep (add 1 second buffer to ensure we are strictly inside the new minute)
        # This prevents running at 14:59:59.999 due to slight clock drift
        time.sleep(sleep_seconds + 1)
        
        # 3. Run the jobs
        run_batch_inference()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Scheduler stopped manually.")
