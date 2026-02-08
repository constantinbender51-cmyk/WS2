import pandas as pd
import numpy as np
from binance import Client
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from flask import Flask, Response
import io

client = Client()

# 1. Fetch hourly → daily OHLC from 2018
klines = client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_1HOUR, '2018-01-01')
df = pd.DataFrame(klines, columns=['ts','o','h','l','c','v','ct','qav','n','T','V','W'])
df['date'] = pd.to_datetime(df['ts'], unit='ms')
df.set_index('date', inplace=True)
df = df.astype(float)
daily = df[['o','h','l','c']].resample('D').last()

# 2. Compute SMAs (in DAYS)
daily['sma_365'] = daily['c'].rolling(365).mean()
daily['sma_730'] = daily['c'].rolling(730).mean()
daily['sma_1460'] = daily['c'].rolling(1460).mean()

# 3. SHIFT BACKWARD (-365 days) → creates GAP at the END (last 365 days = NaN)
daily['sma_365_shifted'] = daily['sma_365'].shift(-365)
daily['sma_730_shifted'] = daily['sma_730'].shift(-730)
daily['sma_1460_shifted'] = daily['sma_1460'].shift(-1460)

# 4. Train RF to fill the END GAP (where shifted SMA is NaN)
def fill_end_gap(df, sma_col, shifted_col):
    # Training data: where shifted SMA exists (middle section)
    train_mask = df[shifted_col].notna()
    # Gap to fill: where shifted SMA is NaN (end of series)
    gap_mask = df[shifted_col].isna()
    
    X_train = df.loc[train_mask, ['o','h','l','c']]
    y_train = df.loc[train_mask, shifted_col]
    
    model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=0, n_jobs=-1)
    model.fit(X_train, y_train)
    
    X_gap = df.loc[gap_mask, ['o','h','l','c']]
    if len(X_gap) > 0:
        df.loc[gap_mask, shifted_col] = model.predict(X_gap)
    
    return df

daily = fill_end_gap(daily, 'sma_365', 'sma_365_shifted')
daily = fill_end_gap(daily, 'sma_730', 'sma_730_shifted')
daily = fill_end_gap(daily, 'sma_1460', 'sma_1460_shifted')

# 5. Plot: OHLC + original SMA (lagging) + shifted+filled SMA (aligned)
def plot():
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # OHLC candles (last 3 years)
    plot_df = daily[-1095:].copy()
    up = plot_df[plot_df.c >= plot_df.o]
    down = plot_df[plot_df.c < plot_df.o]
    
    ax.bar(up.index, up.h - up.l, 0.8, bottom=up.l, color='gray', alpha=0.3)
    ax.bar(up.index, up.c - up.o, 0.8, bottom=up.o, color='green')
    ax.bar(down.index, down.h - down.l, 0.8, bottom=down.l, color='gray', alpha=0.3)
    ax.bar(down.index, down.o - down.c, 0.8, bottom=down.c, color='red')
    
    # Original SMAs (lagging - for reference)
    ax.plot(plot_df.index, plot_df['sma_365'], color='blue', alpha=0.3, linestyle=':', linewidth=1.5, label='365d SMA (lagging)')
    ax.plot(plot_df.index, plot_df['sma_730'], color='green', alpha=0.3, linestyle=':', linewidth=1.5, label='730d SMA (lagging)')
    ax.plot(plot_df.index, plot_df['sma_1460'], color='red', alpha=0.3, linestyle=':', linewidth=1.5, label='1460d SMA (lagging)')
    
    # Shifted + filled SMAs (aligned with price action)
    ax.plot(plot_df.index, plot_df['sma_365_shifted'], color='blue', linewidth=2.5, label='365d SMA (shifted+filled)')
    ax.plot(plot_df.index, plot_df['sma_730_shifted'], color='green', linewidth=2.5, label='730d SMA (shifted+filled)')
    ax.plot(plot_df.index, plot_df['sma_1460_shifted'], color='red', linewidth=2.5, label='1460d SMA (shifted+filled)')
    
    ax.set_title('BTC Daily OHLC + De-Lagged SMAs (Shifted -365/-730/-1460 Days & Filled)', fontsize=16)
    ax.set_ylabel('Price (USDT)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    return buf.getvalue()

# 6. Web server
app = Flask(__name__)

@app.route('/')
def serve():
    return Response(plot(), mimetype='image/png')

if __name__ == '__main__':
    print(f"Data: {len(daily)} days ({daily.index[0].date()} → {daily.index[-1].date()})")
    print("Shifted SMAs backward → filled end gap with Random Forest")
    print("Server: http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, threaded=True)