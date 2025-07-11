import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Technical Analysis
import ta
from datetime import datetime, timedelta
import time

# Page Configuration
st.set_page_config(
    page_title="AI Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.markdown('<h1 class="main-header">🤖 AI Stock Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Advanced Machine Learning Models for Stock Market Prediction")

# Sidebar Configuration
with st.sidebar:
    st.markdown("## 🎯 Configuration")
    
    # Stock Selection - Indian Companies
    popular_stocks = {
        'Reliance Industries': 'RELIANCE.NS',
        'Tata Consultancy Services': 'TCS.NS',
        'Infosys Limited': 'INFY.NS',
        'HDFC Bank': 'HDFCBANK.NS',
        'ICICI Bank': 'ICICIBANK.NS',
        'State Bank of India': 'SBIN.NS',
        'Bharti Airtel': 'BHARTIARTL.NS',
        'ITC Limited': 'ITC.NS',
        'Hindustan Unilever': 'HINDUNILVR.NS',
        'Larsen & Toubro': 'LT.NS',
        'Asian Paints': 'ASIANPAINT.NS',
        'Bajaj Finance': 'BAJFINANCE.NS',
        'Maruti Suzuki': 'MARUTI.NS',
        'Wipro Limited': 'WIPRO.NS',
        'Tech Mahindra': 'TECHM.NS',
        'HCL Technologies': 'HCLTECH.NS',
        'Adani Enterprises': 'ADANIENT.NS',
        'Tata Motors': 'TATAMOTORS.NS',
        'Sun Pharmaceutical': 'SUNPHARMA.NS',
        'Nestle India': 'NESTLEIND.NS'
    }
    
    selected_stock_name = st.selectbox("Select Stock", list(popular_stocks.keys()))
    stock_symbol = popular_stocks[selected_stock_name]
    
    # Time Period
    period_options = {
        '1 Month': '1mo',
        '3 Months': '3mo',
        '6 Months': '6mo',
        '1 Year': '1y',
        '2 Years': '2y',
        '5 Years': '5y'
    }
    
    selected_period = st.selectbox("Historical Data Period", list(period_options.keys()), index=3)
    period = period_options[selected_period]
    
    # Prediction Days
    prediction_days = st.slider("Prediction Days", 1, 90, 30)
    
    # Model Selection
    model_options = {
        'LSTM Neural Network': 'lstm',
        'GRU Neural Network': 'gru',
        'Random Forest': 'rf',
        'Linear Regression': 'lr'
    }
    
    selected_model_name = st.selectbox("ML Model", list(model_options.keys()))
    selected_model = model_options[selected_model_name]
    
    # Advanced Options
    st.markdown("### Advanced Options")
    include_technical_indicators = st.checkbox("Include Technical Indicators", True)
    include_volume = st.checkbox("Include Volume Data", True)
    confidence_interval = st.slider("Confidence Interval (%)", 80, 99, 95)

# Currency Conversion Functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_usd_to_inr_rate():
    """Get current USD to INR exchange rate"""
    try:
        exchange_data = yf.Ticker("USDINR=X")
        rate = exchange_data.history(period="1d")['Close'].iloc[-1]
        return rate
    except:
        return 83.0  # Fallback rate

def convert_to_inr(usd_amount):
    """Convert USD to INR"""
    rate = get_usd_to_inr_rate()
    return usd_amount * rate

def format_inr_currency(amount):
    """Format currency in Indian style (Lakhs/Crores)"""
    if amount >= 10000000:  # 1 Crore
        return f"₹{amount/10000000:.2f}Cr"
    elif amount >= 100000:  # 1 Lakh
        return f"₹{amount/100000:.2f}L"
    else:
        return f"₹{amount:.2f}"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_stock_data(symbol, period):
    """Load stock data using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def add_technical_indicators(data):
    """Add technical indicators to the dataset"""
    # Moving Averages
    data['MA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['MA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
    data['EMA_12'] = ta.trend.ema_indicator(data['Close'], window=12)
    data['EMA_26'] = ta.trend.ema_indicator(data['Close'], window=26)
    
    # RSI
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    
    # MACD
    data['MACD'] = ta.trend.macd_diff(data['Close'])
    
    # Bollinger Bands
    data['BB_Upper'] = ta.volatility.bollinger_hband(data['Close'])
    data['BB_Lower'] = ta.volatility.bollinger_lband(data['Close'])
    
    # Volatility
    data['Volatility'] = data['Close'].pct_change().rolling(window=20).std()
    
    return data

def prepare_lstm_data(data, look_back=60):
    """Prepare data for LSTM model"""
    if data.empty:
        st.warning("Input data for LSTM preparation is empty.")
        return np.array([]), np.array([]), MinMaxScaler() # Return empty arrays and a scaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    X, y = [], []
    
    # Crucial check: Ensure enough data for the look_back period
    if len(scaled_data) <= look_back:
        st.warning(f"Not enough historical data ({len(scaled_data)} points) for a look-back period of {look_back} days. Please select a longer data period (e.g., '1 Year' or more) to use LSTM/GRU models effectively.")
        return np.array([]), np.array([]), scaler # Return empty numpy arrays if data is insufficient
    
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    """Build LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def build_gru_model(input_shape):
    """Build GRU model"""
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(50, return_sequences=True),
        Dropout(0.2),
        GRU(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def calculate_metrics(actual, predicted):
    """Calculate prediction metrics"""
    # Check if actual or predicted are empty, or if they have non-comparable sizes
    if len(actual) == 0 or len(predicted) == 0 or len(actual) != len(predicted):
        st.warning("Cannot calculate metrics: Actual or predicted values are empty or have mismatched lengths.")
        return {
            'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R²': np.nan, 'Accuracy': np.nan
        }

    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    
    # Handle case where r2_score might fail (e.g., if all actual values are the same)
    try:
        r2 = r2_score(actual, predicted)
    except ValueError:
        r2 = 0.0 # Or raise a specific error, depending on desired behavior
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Accuracy': max(0, r2 * 100)
    }

# Main Application
def main():
    # Load Data
    with st.spinner(f"Loading {selected_stock_name} data..."):
        data = load_stock_data(stock_symbol, period)
    
    if data is None or data.empty:
        st.error("Failed to load stock data. Please try again or choose a different stock/period.")
        return
    
    # Add technical indicators if selected
    if include_technical_indicators:
        initial_data_rows = len(data)
        data = add_technical_indicators(data)
        # Drop NaN values introduced by technical indicators
        data.dropna(inplace=True)
        if len(data) == 0:
            st.error(f"After adding technical indicators, no valid data points remain. Please choose a longer historical period (e.g., '1 Year' or more). Original rows: {initial_data_rows}")
            return
        elif len(data) < initial_data_rows:
            st.warning(f"Some data points were removed due to NaN values after adding technical indicators. Data points remaining: {len(data)}")

    # Display Stock Information
    # Ensure data has enough rows before accessing iloc[-1] and iloc[-2]
    if len(data) < 2:
        st.warning("Not enough historical data to display current price and daily change. Please select a longer period.")
        current_price = data['Close'].iloc[-1] if not data.empty else 0
        price_change = 0
        price_change_pct = 0
    else:
        current_price = data['Close'].iloc[-1]
        price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
        price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{format_inr_currency(current_price)}</h3>
            <p>Current Price</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = "green" if price_change >= 0 else "red"
        arrow = "↗️" if price_change >= 0 else "↘️"
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {color};">{arrow} {format_inr_currency(abs(price_change))}</h3>
            <p>Daily Change</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{price_change_pct:.2f}%</h3>
            <p>Change %</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_volume = data['Volume'].tail(30).mean()
        if pd.isna(avg_volume): # Handle case where tail(30) is empty or all NaN
            volume_text = "N/A"
        elif avg_volume >= 10000000:  # 1 Crore
            volume_text = f"{avg_volume/10000000:.1f}Cr"
        elif avg_volume >= 100000:  # 1 Lakh
            volume_text = f"{avg_volume/100000:.1f}L"
        else:
            volume_text = f"{avg_volume:.0f}"
        st.markdown(f"""
        <div class="metric-card">
            <h3>{volume_text}</h3>
            <p>Avg Volume (30d)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Historical Data Visualization
    st.markdown("## 📊 Historical Data Analysis")
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Stock Price', 'Volume', 'Technical Indicators'),
        vertical_spacing=0.1,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price Chart
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue')),
        row=1, col=1
    )
    
    if include_technical_indicators:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MA_20'], name='MA 20', line=dict(color='orange')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MA_50'], name='MA 50', line=dict(color='red')),
            row=1, col=1
        )
    
    # Volume Chart
    if 'Volume' in data.columns and not data['Volume'].empty:
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'),
            row=2, col=1
        )
    else:
        st.warning("Volume data is not available or is empty.")

    # Technical Indicators
    if include_technical_indicators:
        if 'RSI' in data.columns and not data['RSI'].empty:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
                row=3, col=1
            )
        else:
            st.warning("RSI data could not be computed or is empty.")
    
    fig.update_layout(height=800, title_text=f"{selected_stock_name} Stock Analysis (NSE)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Model Training and Prediction
    st.markdown("## 🤖 AI Model Prediction")
    
    if st.button("🚀 Generate Predictions", type="primary"):
        with st.spinner("Training AI model and generating predictions..."):
            progress_bar = st.progress(0)
            
            # Prepare data
            close_prices = data['Close'].dropna()

            if close_prices.empty:
                st.error("Cannot proceed with prediction: 'Close' price data is empty after dropping NaNs.")
                progress_bar.empty()
                return

            metrics = {} # Initialize metrics dictionary

            if selected_model in ['lstm', 'gru']:
                X, y, scaler = prepare_lstm_data(close_prices)

                # Check if prepare_lstm_data returned empty arrays
                if X.size == 0:
                    progress_bar.empty()
                    return # Error message already shown in prepare_lstm_data
                
                # Split data
                # Ensure enough data points for a meaningful split
                if len(X) < 2: # Need at least two data points for train/test split
                    st.error("Not enough data points to create training and testing sets for the selected model. Please try a longer 'Historical Data Period'.")
                    progress_bar.empty()
                    return

                split = int(0.8 * len(X))
                
                # Ensure neither train nor test set is empty after split
                if split == 0 or split == len(X):
                    st.error("Train or test set is empty after splitting data. Adjust 'Historical Data Period' to ensure enough data.")
                    progress_bar.empty()
                    return

                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]
                
                # Reshape for LSTM/GRU
                # Check if X_train is empty *before* reshaping, as reshaping an empty array can be tricky
                if X_train.shape[0] == 0:
                    st.error("Training data (X_train) is empty after split. Please ensure enough historical data is selected.")
                    progress_bar.empty()
                    return

                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                
                # Check X_test as well, it might be empty if the split results in a very small test set
                test_predictions = np.array([]) # Initialize as empty
                y_test_actual = np.array([]) # Initialize as empty

                if X_test.shape[0] > 0: # Only reshape and predict if X_test is not empty
                    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                else:
                    st.warning("Test data (X_test) is empty. Model performance metrics will not be calculated. Consider a longer historical period for better evaluation.")
                
                progress_bar.progress(30)
                
                # Build and train model
                if selected_model == 'lstm':
                    model = build_lstm_model((X_train.shape[1], 1))
                else: # GRU
                    model = build_gru_model((X_train.shape[1], 1))

                model.fit(X_train, y_train, epochs=50, batch_size=32, 
                                  validation_split=0.1, verbose=0)
                
                progress_bar.progress(70)
                
                # Make predictions
                # Only predict on test if X_test is not empty
                if X_test.shape[0] > 0:
                    test_predictions = model.predict(X_test)
                    test_predictions = scaler.inverse_transform(test_predictions)
                    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                else:
                    test_predictions = np.array([])
                    y_test_actual = np.array([])

                # Future predictions
                # Ensure `last_sequence` is valid for prediction
                if X_test.shape[0] > 0:
                    last_sequence = X_test[-1:].reshape(1, X_test.shape[1], 1)
                elif X_train.shape[0] > 0: # Fallback to training data if test is empty
                    last_sequence = X_train[-1:].reshape(1, X_train.shape[1], 1)
                else:
                    st.error("No historical data available to generate future predictions. Please increase the data period.")
                    progress_bar.empty()
                    return np.array([]) # Return empty array for future predictions

                future_predictions = []
                
                for _ in range(prediction_days):
                    pred = model.predict(last_sequence, verbose=0)
                    future_predictions.append(pred[0, 0])
                    
                    # Update sequence
                    last_sequence = np.append(last_sequence[:, 1:, :], 
                                            pred.reshape(1, 1, 1), axis=1)
                
                future_predictions = scaler.inverse_transform(
                    np.array(future_predictions).reshape(-1, 1)
                ).flatten()
                
                progress_bar.progress(100)
                
                # Calculate metrics only if test data was available
                if y_test_actual.size > 0 and test_predictions.size > 0:
                    metrics = calculate_metrics(y_test_actual.flatten(), test_predictions.flatten())
                else:
                    st.warning("Skipping metric calculation due to insufficient test data.")
                    metrics = {
                        'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R²': np.nan, 'Accuracy': np.nan
                    }

            elif selected_model == 'rf':
                # Random Forest Model
                features = pd.DataFrame()
                features['Close'] = close_prices
                features['Close_lag1'] = close_prices.shift(1)
                features['Close_lag2'] = close_prices.shift(2)
                features['Close_lag3'] = close_prices.shift(3)
                features['MA_5'] = close_prices.rolling(5).mean()
                features['MA_10'] = close_prices.rolling(10).mean()
                features['Returns'] = close_prices.pct_change()
                
                if include_technical_indicators:
                    # Ensure data has enough rows for indicators after initial dropna
                    if 'RSI' in data.columns: features['RSI'] = data['RSI']
                    if 'MACD' in data.columns: features['MACD'] = data['MACD']
                
                features = features.dropna() # Drop NaNs introduced by shifts/rolling means/indicators
                
                if features.empty:
                    st.error("Not enough historical data to create features for Random Forest. Please select a longer data period.")
                    progress_bar.empty()
                    return

                X = features.drop('Close', axis=1)
                y = features['Close']
                
                if X.empty:
                    st.error("Features (X) for Random Forest are empty after preparation. Cannot train model.")
                    progress_bar.empty()
                    return

                # Split data
                if len(X) < 2:
                    st.error("Not enough data points for training Random Forest. Please choose a longer historical period.")
                    progress_bar.empty()
                    return

                split = int(0.8 * len(X))
                if split == 0 or split == len(X):
                    st.error("Train or test set is empty after splitting data for Random Forest. Adjust 'Historical Data Period'.")
                    progress_bar.empty()
                    return

                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]

                if X_train.empty:
                    st.error("Training data (X_train) for Random Forest is empty after split. Please ensure enough historical data.")
                    progress_bar.empty()
                    return
                
                progress_bar.progress(50)
                
                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Predictions
                test_predictions = np.array([])
                if not X_test.empty:
                    test_predictions = model.predict(X_test)
                else:
                    st.warning("Test data (X_test) for Random Forest is empty. Model performance metrics will not be calculated.")

                # Future predictions
                future_predictions = []
                last_features = X.iloc[-1:].copy()
                
                if last_features.empty:
                    st.error("Cannot generate future predictions for Random Forest: last features are empty.")
                    progress_bar.empty()
                    return

                for _ in range(prediction_days):
                    pred = model.predict(last_features)[0]
                    future_predictions.append(pred)
                    
                    # Update features for next prediction
                    # Ensure columns exist before accessing
                    if 'Close_lag3' in last_features.columns:
                        last_features['Close_lag3'] = last_features['Close_lag2'].iloc[0] if 'Close_lag2' in last_features.columns else pred
                    if 'Close_lag2' in last_features.columns:
                        last_features['Close_lag2'] = last_features['Close_lag1'].iloc[0] if 'Close_lag1' in last_features.columns else pred
                    if 'Close_lag1' in last_features.columns:
                        last_features['Close_lag1'] = pred
                    
                    # Recalculate MA_5 based on newly predicted values
                    # This is a simplified approach; more robust would use a sliding window of actuals + predictions
                    if 'MA_5' in last_features.columns:
                        temp_series = pd.Series([
                            last_features['Close_lag1'].iloc[0] if 'Close_lag1' in last_features.columns else pred,
                            last_features['Close_lag2'].iloc[0] if 'Close_lag2' in last_features.columns else pred,
                            last_features['Close_lag3'].iloc[0] if 'Close_lag3' in last_features.columns else pred,
                            # Need more points for true MA_5, simplifying for example
                            pred, pred # Placeholder, in a real scenario you'd need the 2 prior actuals
                        ])
                        last_features['MA_5'] = temp_series.tail(3).mean() # Adjust to be realistic
                    
                    if 'Returns' in last_features.columns and 'Close_lag1' in last_features.columns:
                         last_features['Returns'] = (pred - last_features['Close_lag1'].iloc[0]) / last_features['Close_lag1'].iloc[0]
                    
                    # Handle technical indicators if included (simplified as they depend on actual historical data)
                    if include_technical_indicators:
                        # For future predictions, technical indicators are hard to calculate accurately without actual future data.
                        # For simplicity, they might be kept constant or estimated.
                        # For a robust solution, you'd need a more complex way to project indicators.
                        pass # Keeping them as they were, or using a simpler projection

                progress_bar.progress(100)
                
                if len(y_test) > 0 and len(test_predictions) > 0:
                    metrics = calculate_metrics(y_test.values, test_predictions)
                else:
                    st.warning("Skipping metric calculation due to insufficient test data for Random Forest.")
                    metrics = {
                        'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R²': np.nan, 'Accuracy': np.nan
                    }
                
            else:  # Linear Regression
                X_lr = np.arange(len(close_prices)).reshape(-1, 1)
                y_lr = close_prices.values
                
                if len(X_lr) < 2:
                    st.error("Not enough data points for Linear Regression. Please choose a longer historical period.")
                    progress_bar.empty()
                    return

                split = int(0.8 * len(X_lr))
                if split == 0 or split == len(X_lr):
                    st.error("Train or test set is empty after splitting data for Linear Regression. Adjust 'Historical Data Period'.")
                    progress_bar.empty()
                    return

                X_train, X_test = X_lr[:split], X_lr[split:]
                y_train, y_test = y_lr[:split], y_lr[split:]

                if X_train.shape[0] == 0:
                    st.error("Training data (X_train) for Linear Regression is empty after split. Please ensure enough historical data.")
                    progress_bar.empty()
                    return

                progress_bar.progress(50)
                
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                test_predictions = np.array([])
                if X_test.shape[0] > 0:
                    test_predictions = model.predict(X_test)
                else:
                    st.warning("Test data (X_test) for Linear Regression is empty. Model performance metrics will not be calculated.")
                
                # Future predictions
                future_X = np.arange(len(close_prices), 
                                   len(close_prices) + prediction_days).reshape(-1, 1)
                future_predictions = model.predict(future_X)
                
                progress_bar.progress(100)
                
                if len(y_test) > 0 and len(test_predictions) > 0:
                    metrics = calculate_metrics(y_test, test_predictions)
                else:
                    st.warning("Skipping metric calculation due to insufficient test data for Linear Regression.")
                    metrics = {
                        'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R²': np.nan, 'Accuracy': np.nan
                    }
            
            progress_bar.empty()
            
            # Display Results
            st.markdown("### 📈 Prediction Results")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics.get('Accuracy', np.nan):.2f}%" if not pd.isna(metrics.get('Accuracy')) else "N/A")
            with col2:
                st.metric("RMSE", f"{metrics.get('RMSE', np.nan):.2f}" if not pd.isna(metrics.get('RMSE')) else "N/A")
            with col3:
                st.metric("MAE", f"{metrics.get('MAE', np.nan):.2f}" if not pd.isna(metrics.get('MAE')) else "N/A")
            with col4:
                st.metric("R² Score", f"{metrics.get('R²', np.nan):.4f}" if not pd.isna(metrics.get('R²')) else "N/A")
            
            # Prediction Chart
            future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), 
                                       periods=prediction_days, freq='D')
            
            fig_pred = go.Figure()
            
            # Historical data
            fig_pred.add_trace(
                go.Scatter(x=data.index[-100:], y=data['Close'][-100:], 
                          name='Historical', line=dict(color='blue'))
            )
            
            # Predictions
            if len(future_predictions) > 0:
                fig_pred.add_trace(
                    go.Scatter(x=future_dates, y=future_predictions, 
                              name='Predictions', line=dict(color='red', dash='dash'))
                )
                
                # Add confidence interval
                confidence_lower = future_predictions * (1 - (100 - confidence_interval) / 200)
                confidence_upper = future_predictions * (1 + (100 - confidence_interval) / 200)
                
                fig_pred.add_trace(
                    go.Scatter(x=future_dates, y=confidence_upper, 
                              fill=None, mode='lines', line_color='rgba(0,0,0,0)', 
                              showlegend=False)
                )
                fig_pred.add_trace(
                    go.Scatter(x=future_dates, y=confidence_lower, 
                              fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', 
                              name=f'{confidence_interval}% Confidence Interval', 
                              fillcolor='rgba(255,0,0,0.2)')
                )
            else:
                st.warning("No future predictions generated to display.")
            
            fig_pred.update_layout(
                title=f"{selected_stock_name} - {prediction_days} Day Prediction (NSE)",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=500
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Prediction Summary
            current_price = data['Close'].iloc[-1] if not data.empty else 0
            predicted_price = future_predictions[-1] if len(future_predictions) > 0 else current_price
            
            if current_price != 0:
                price_change = predicted_price - current_price
                price_change_pct = (price_change / current_price) * 100
            else:
                price_change = 0
                price_change_pct = 0
            
            st.markdown(f"""
            <div class="prediction-card">
                <h3>🎯 Prediction Summary</h3>
                <p><strong>Current Price:</strong> {format_inr_currency(current_price)}</p>
                <p><strong>Predicted Price ({prediction_days} days):</strong> {format_inr_currency(predicted_price)}</p>
                <p><strong>Expected Change:</strong> {format_inr_currency(abs(price_change))} ({price_change_pct:+.2f}%)</p>
                <p><strong>Model:</strong> {selected_model_name}</p>
                <p><strong>Confidence:</strong> {confidence_interval}%</p>
                <p><strong>Exchange:</strong> NSE (National Stock Exchange)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature Importance (for Random Forest)
            if selected_model == 'rf':
                st.markdown("### 🔍 Feature Importance")
                if 'model' in locals() and hasattr(model, 'feature_importances_') and not X.empty:
                    feature_importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    fig_importance = px.bar(
                        feature_importance, 
                        x='importance', 
                        y='feature',
                        title='Feature Importance in Random Forest Model',
                        orientation='h'
                    )
                    
                    st.plotly_chart(fig_importance, use_container_width=True)
                else:
                    st.info("Feature importance is available for Random Forest models, but could not be computed due to insufficient data or model state.")
    
    # Additional Information
    st.markdown("## ℹ️ Model Information & Market Context")
    
    # Add Indian market context
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏦 Indian Stock Market Info")
        st.info(f"""
        **Exchange:** NSE (National Stock Exchange of India)
        **Currency:** Indian Rupee (₹)
        **Market Hours:** 9:15 AM - 3:30 PM IST
        **Current USD/INR Rate:** ₹{get_usd_to_inr_rate():.2f}
        """)
    
    with col2:
        st.markdown("### 📊 Market Indices")
        st.info("""
        **NIFTY 50:** Top 50 companies by market cap
        **SENSEX:** BSE benchmark index
        **NIFTY Bank:** Banking sector index
        **NIFTY IT:** Information Technology sector
        """)
    
    st.markdown("## ℹ️ Model Information")
    
    with st.expander("About the Models"):
        st.markdown("""
        **LSTM (Long Short-Term Memory):** Deep learning model excellent for time series prediction.
        Captures long-term dependencies in stock price movements.
        
        **GRU (Gated Recurrent Unit):** Simplified version of LSTM with fewer parameters.
        Often performs comparably to LSTM with faster training.
        
        **Random Forest:** Ensemble method that combines multiple decision trees.
        Good for capturing non-linear patterns and feature interactions.
        
        **Linear Regression:** Simple baseline model that assumes linear relationship.
        Useful for identifying overall trends.
        """)
    
    with st.expander("Disclaimer"):
        st.markdown("""
        ⚠️ **Important Disclaimer:**
        
        This application is for educational and research purposes only. 
        Indian stock market predictions are inherently uncertain and past performance 
        does not guarantee future results. Always consult with a SEBI-registered 
        financial advisor before making investment decisions.
        
        The predictions generated by this tool should not be considered as 
        financial advice or recommendations to buy or sell securities listed 
        on NSE/BSE. Please be aware of market risks and invest according to 
        your risk appetite.
        
        **Important:** Consider factors like market volatility, economic conditions, 
        company fundamentals, and regulatory changes before making investment decisions.
        """)
    
    # Add market timings info
    st.markdown("### 🕐 Market Timings (IST)")
    st.info("""
    **Pre-market:** 9:00 AM - 9:15 AM
    **Regular Trading:** 9:15 AM - 3:30 PM  
    **Post-market:** 3:40 PM - 4:00 PM
    **Currency:** Monday - Friday (9:00 AM - 5:00 PM)
    """)

if __name__ == "__main__":
    main()