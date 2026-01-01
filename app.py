import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px
import gdown
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Railway LSTM Deployment",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .main .block-container { padding-top: 2rem; }
    h1 { color: #4da6ff; }
    h2 { border-bottom: 1px solid #333; padding-bottom: 10px; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_data
def load_data(source_type, file_upload, url_input):
    df = None
    try:
        if source_type == "Upload CSV":
            if file_upload is not None:
                df = pd.read_csv(file_upload)
        elif source_type == "Google Drive / URL":
            if url_input:
                if "drive.google.com" in url_input:
                    output = 'downloaded_data.csv'
                    # Quietly download
                    gdown.download(url_input, output, quiet=True, fuzzy=True)
                    df = pd.read_csv(output)
                else:
                    df = pd.read_csv(url_input)
        elif source_type == "Generate Sample Data":
            # Generate synthetic railway track temperature data
            date_rng = pd.date_range(start='1/1/2024', end='1/08/2024', freq='T')
            val = 20 + np.random.randn(len(date_rng)).cumsum() * 0.5 # Random walk
            seasonality = np.sin(np.arange(len(date_rng)) * (2 * np.pi / (24*60))) * 10 # Daily cycle
            df = pd.DataFrame({'timestamp': date_rng, 'sensor_value': val + seasonality})
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# --- Main Application Layout ---

st.title("🚄 Railway Deployment LSTM Framework")
st.markdown("Generative Time-Series Forecasting with Confidence Estimation")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("1. Data Configuration")
    data_source = st.selectbox("Data Source", ["Generate Sample Data", "Upload CSV", "Google Drive / URL"])
    
    uploaded_file = None
    url_input = ""
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    elif data_source == "Google Drive / URL":
        url_input = st.text_input("Enter URL (Direct or GDrive share link)")
        
    st.header("2. Preprocessing")
    resample_freq = st.number_input("Resample Frequency (Minutes)", min_value=1, value=15, step=1)
    split_ratio = st.slider("Train/Validation Split", 0.5, 0.95, 0.8)
    
    st.header("3. Model Architecture")
    n_layers = st.number_input("Number of LSTM Layers", 1, 5, 2)
    units = st.number_input("Units per Layer", 10, 500, 50)
    dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2)
    
    st.header("4. Hyperparameters")
    look_back = st.number_input("Lookback Window (Steps)", 1, 100, 12)
    epochs = st.number_input("Epochs", 1, 200, 20)
    batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")

    train_btn = st.button("🚀 Train Model")

# --- Main Content Area ---

# 1. Load Data
df = load_data(data_source, uploaded_file, url_input)

if df is not None:
    # Column Selection
    cols = df.columns.tolist()
    
    c1, c2 = st.columns(2)
    with c1:
        date_col = st.selectbox("Select Timestamp Column", cols, index=0 if 'timestamp' in cols else 0)
    with c2:
        target_col = st.selectbox("Select Target Variable", cols, index=1 if 'value' in cols or 'sensor_value' in cols else min(1, len(cols)-1))
    
    # Preprocessing
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)
        df_resampled = df.set_index(date_col).resample(f'{resample_freq}T').mean().dropna()
        
        st.subheader("Data Preview")
        with st.expander("Raw Data vs Resampled Data", expanded=True):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.dataframe(df_resampled.head(), height=250)
                st.caption(f"Original: {len(df)} rows | Resampled: {len(df_resampled)} rows")
            with col2:
                fig_preview = px.line(df_resampled, y=target_col, title=f"{target_col} over Time ({resample_freq} min avg)")
                fig_preview.update_layout(margin=dict(l=0,r=0,t=30,b=0), height=250)
                st.plotly_chart(fig_preview, use_container_width=True)
                
    except Exception as e:
        st.error(f"Preprocessing Error: {e}. Please ensure the timestamp column is valid.")
        st.stop()
else:
    st.info("Please select a data source to begin.")
    st.stop()


# 2. Training Logic
if train_btn:
    st.divider()
    st.subheader("Training Live Feed")
    
    # Data Preparation
    data_values = df_resampled[target_col].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data_values)
    
    train_size = int(len(dataset) * split_ratio)
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    
    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back)
    
    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Model Building
    model = Sequential()
    for i in range(n_layers):
        return_seq = True if i < n_layers - 1 else False
        if i == 0:
            model.add(LSTM(units, input_shape=(look_back, 1), return_sequences=return_seq))
        else:
            model.add(LSTM(units, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    # Custom Callback for Streamlit Live Updates
    progress_bar = st.progress(0)
    status_text = st.empty()
    chart_placeholder = st.empty()
    
    class StreamlitCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            self.history = {'loss': [], 'val_loss': []}
            
        def on_epoch_end(self, epoch, logs=None):
            self.history['loss'].append(logs['loss'])
            self.history['val_loss'].append(logs['val_loss'])
            
            # Update Progress
            prog = (epoch + 1) / epochs
            progress_bar.progress(prog)
            status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {logs['loss']:.5f} - Val Loss: {logs['val_loss']:.5f}")
            
            # Live Chart Update
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=self.history['loss'], mode='lines', name='Train Loss'))
            fig_loss.add_trace(go.Scatter(y=self.history['val_loss'], mode='lines', name='Val Loss'))
            fig_loss.update_layout(
                title="Training Progress (Loss)", 
                xaxis_title="Epoch", 
                yaxis_title="MSE Loss",
                height=300,
                margin=dict(l=0,r=0,t=30,b=0)
            )
            chart_placeholder.plotly_chart(fig_loss, use_container_width=True)

    # Train
    with st.spinner('Training model...'):
        history = model.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, Y_test),
            verbose=0,
            callbacks=[StreamlitCallback()]
        )
    
    st.success("Training Complete!")
    
    # 3. Predictions & Evaluation
    st.divider()
    st.subheader("Model Evaluation & Forecasting")
    
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Invert predictions
    train_predict = scaler.inverse_transform(train_predict)
    Y_train_inv = scaler.inverse_transform([Y_train])
    test_predict = scaler.inverse_transform(test_predict)
    Y_test_inv = scaler.inverse_transform([Y_test])
    
    # Metrics
    mse = mean_squared_error(Y_test_inv[0], test_predict[:,0])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test_inv[0], test_predict[:,0])
    
    # Metrics Display
    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE", f"{rmse:.4f}")
    m2.metric("MSE", f"{mse:.4f}")
    m3.metric("MAE", f"{mae:.4f}")
    
    # Confidence Score / Interval Calculation
    # We use the standard deviation of residuals on the test set to define a confidence band
    residuals = Y_test_inv[0] - test_predict[:,0]
    std_resid = np.std(residuals)
    confidence_interval = 1.96 * std_resid  # 95% confidence assuming normal distribution of errors
    
    # Preparing Data for Final Plot
    # Shift train predictions for plotting
    train_plot_data = np.empty_like(dataset)
    train_plot_data[:, :] = np.nan
    train_plot_data[look_back:len(train_predict)+look_back, :] = train_predict
    
    # Shift test predictions for plotting
    test_plot_data = np.empty_like(dataset)
    test_plot_data[:, :] = np.nan
    # Correct indexing for test data placement
    test_start_idx = len(train_predict) + (look_back * 2) + 1
    end_idx = min(len(dataset), test_start_idx + len(test_predict))
    # We create a specific array for the test period to make plotting easier
    
    # Reconstruct DataFrame for Plotly
    # We will plot the last N points to keep it readable, or full dataset
    
    full_original = scaler.inverse_transform(dataset)
    
    # Align dates
    dates = df_resampled.index
    
    # Create a clean DataFrame for visualization
    res_df = pd.DataFrame(index=dates)
    res_df['Actual'] = full_original
    
    # Map predictions to timestamps
    # Train
    train_dates = dates[look_back : look_back + len(train_predict)]
    train_series = pd.Series(train_predict.flatten(), index=train_dates)
    
    # Test
    # The offset for test depends on the split
    test_start = len(train) + look_back
    if test_start < len(dates):
        test_dates = dates[test_start : test_start + len(test_predict)]
        test_series = pd.Series(test_predict.flatten(), index=test_dates)
    
        # Visualization
        fig_res = go.Figure()
        
        # Actual Data
        fig_res.add_trace(go.Scatter(
            x=res_df.index, y=res_df['Actual'],
            mode='lines', name='Actual Data',
            line=dict(color='gray', width=1)
        ))
        
        # Train Predictions
        fig_res.add_trace(go.Scatter(
            x=train_series.index, y=train_series.values,
            mode='lines', name='Train Predictions',
            line=dict(color='#3b82f6')
        ))
        
        # Test Predictions & Confidence
        fig_res.add_trace(go.Scatter(
            x=test_series.index, y=test_series.values,
            mode='lines', name='Test Predictions',
            line=dict(color='#10b981')
        ))
        
        # Confidence Band (Upper/Lower)
        # Only for test data to show forecasting uncertainty
        upper_bound = test_series.values + confidence_interval
        lower_bound = test_series.values - confidence_interval
        
        fig_res.add_trace(go.Scatter(
            x=test_series.index, 
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            name='Upper Confidence'
        ))
        
        fig_res.add_trace(go.Scatter(
            x=test_series.index, 
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fill='tonexty', # Fill to upper bound
            fillcolor='rgba(16, 185, 129, 0.2)',
            showlegend=True,
            name=f'95% Confidence (±{confidence_interval:.2f})'
        ))

        fig_res.update_layout(
            title="Railway Sequence Prediction",
            xaxis_title="Timestamp",
            yaxis_title=target_col,
            height=600,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_res, use_container_width=True)
        
        # Prediction Table
        with st.expander("View Detailed Prediction Data"):
            results_table = pd.DataFrame({
                "Timestamp": test_dates,
                "Actual": Y_test_inv.flatten(),
                "Predicted": test_predict.flatten(),
                "Error": (Y_test_inv.flatten() - test_predict.flatten())
            })
            st.dataframe(results_table)