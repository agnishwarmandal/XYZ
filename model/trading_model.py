import os
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
import sys
import traceback

# Manually add TA-Lib library path
ta_lib_path = 'C:\\ta-lib\\lib'
if ta_lib_path not in sys.path:
    sys.path.append(ta_lib_path)
    os.environ['PATH'] += f';{ta_lib_path}'

try:
    import talib
except ImportError:
    print("Warning: Could not import TA-Lib. Technical indicators might not work.")
    talib = None

import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import logging
import sys
import os

def calculate_rsi(data, periods=14):
    """
    Calculate Relative Strength Index (RSI) without TA-Lib
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    
    rs = gain / loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD) without TA-Lib
    """
    exp1 = data.ewm(span=fast_period, adjust=False).mean()
    exp2 = data.ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def calculate_bollinger_bands(data, periods=20, std_dev=2):
    """
    Calculate Bollinger Bands without TA-Lib
    """
    rolling_mean = data.rolling(window=periods).mean()
    rolling_std = data.rolling(window=periods).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band, rolling_mean, lower_band

def calculate_stochastic_oscillator(high, low, close, k_period=14, d_period=3):
    """
    Calculate Stochastic Oscillator without TA-Lib
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    return stoch_k, stoch_d

class NiftyTradingModel:
    def __init__(self):
        """
        Initialize Nifty Intraday Trading Model
        """
        # Configure logging with UTF-8 encoding
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s:%(name)s:%(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Ensure proper encoding for stdout/stderr
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
        if sys.stderr.encoding != 'utf-8':
            sys.stderr.reconfigure(encoding='utf-8')
        
        # Model configuration
        self.scaler = MinMaxScaler()
        self.feature_columns = []
        self.lookback_period = 30  # Reduced for intraday trading
        self.model = None
        
        # Intraday trading parameters
        self.trading_start_time = pd.Timestamp('09:15:00').time()
        self.trading_end_time = pd.Timestamp('15:30:00').time()
        
        # Risk management
        self.max_daily_trades = 5
        self.stop_loss_percentage = 0.5  # 0.5% stop loss
        self.take_profit_percentage = 1.0  # 1% take profit

    def download_intraday_data(self, date=None):
        """
        Download intraday minute-level data for Nifty 50
        
        Args:
            date (str, optional): Specific date for intraday data
        
        Returns:
            pd.DataFrame: Intraday trading data
        """
        try:
            # Use provided date or today
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')
                
            # Download data with multiple interval fallback
            intervals = ['1m', '2m', '5m']
            data = None
            
            for interval in intervals:
                try:
                    # Download data
                    ticker = yf.Ticker('^NSEI')
                    data = ticker.history(period='1d', interval=interval)
                    
                    if len(data) > 0:
                        self.logger.info(f"Downloaded {len(data)} data points using {interval} interval")
                        break
                        
                except Exception as e:
                    self.logger.warning(f"Failed to download data with {interval} interval: {str(e)}")
                    continue
            
            if data is None or len(data) == 0:
                raise Exception("Failed to download data with all intervals")
                
            # Filter trading hours (9:15 AM to 3:30 PM IST)
            data.index = pd.to_datetime(data.index)
            market_open = pd.Timestamp(date + ' 09:15:00+05:30')
            market_close = pd.Timestamp(date + ' 15:30:00+05:30')
            
            data = data.loc[market_open:market_close]
            
            self.logger.info(f"Initial dataframe shape: {data.shape}")
            self.logger.info(f"Initial dataframe columns: {list(data.columns)}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Data download failed: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def _filter_trading_hours(self, df):
        """
        Filter dataframe to include only Indian market trading hours
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        try:
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Define trading hours (9:15 AM to 3:30 PM IST)
            df = df.between_time('09:15', '15:30')
            
            return df
        
        except Exception as e:
            self.logger.error(f"Trading hours filtering error: {e}")
            return df

    def _engineer_features(self, data):
        """
        Enhanced feature engineering with technical indicators
        
        Args:
            data (pd.DataFrame): Raw price data
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        try:
            self.logger.info(f"Initial dataframe shape: {data.shape}")
            self.logger.info(f"Initial dataframe columns: {list(data.columns)}")
            
            # Basic features
            data['Returns'] = data['Close'].pct_change()
            
            # Volume features
            data['Volume_EMA'] = data['Volume'].ewm(span=10, adjust=False).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_EMA']
            
            # Price momentum features
            data['Price_EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
            data['Price_EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
            data['Price_EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
            data['Price_EMA_Diff_5_10'] = data['Price_EMA_5'] - data['Price_EMA_10']
            data['Price_EMA_Diff_10_20'] = data['Price_EMA_10'] - data['Price_EMA_20']
            
            # Fallback technical indicators
            # Fallback calculations without TA-Lib
            data['RSI'] = calculate_rsi(data['Close'], periods=14)
            
            data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = calculate_macd(
                data['Close'], fast_period=12, slow_period=26, signal_period=9
            )
            
            data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = calculate_bollinger_bands(
                data['Close'], periods=20, std_dev=2
            )
            data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
            
            data['Stoch_K'], data['Stoch_D'] = calculate_stochastic_oscillator(
                data['High'], data['Low'], data['Close'], 
                k_period=14, d_period=3
            )
            
            # Volatility and trend indicators
            data['ATR'] = data['High'] - data['Low']
            data['Volatility_Ratio'] = data['ATR'] / data['Close']
            
            # Simplified ROC
            data['ROC'] = data['Close'].pct_change(periods=10)
            
            # Simplified MFI
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            money_flow = typical_price * data['Volume']
            data['MFI'] = money_flow.rolling(window=14).mean()
            
            # Simplified ADX and CCI
            data['ADX'] = abs(data['High'] - data['Low'].shift(1)) / data['Close']
            data['CCI'] = (data['Close'] - data['Close'].rolling(window=14).mean()) / (0.015 * data['Close'].rolling(window=14).std())
            
            # Drop initial rows with NaN
            data.dropna(inplace=True)
            
            self.logger.info(f"Prepared {len(data)} valid feature rows")
            self.logger.info(f"Feature columns: {list(data.columns)}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}")
            self.logger.error(f"Error details: {traceback.format_exc()}")
            return pd.DataFrame()

    def generate_intraday_signals(self, data):
        """
        Generate intraday trading signals based on multiple technical indicators
        
        Args:
            data (pd.DataFrame): DataFrame with technical indicators
            
        Returns:
            int: Trading signal (-1: Sell, 0: Hold, 1: Buy)
        """
        try:
            # Extract latest values
            latest = data.iloc[-1]
            
            # Initialize signal as hold (0)
            signal = 0
            
            # Comprehensive buy conditions
            buy_conditions = [
                # Strong momentum indicators
                (latest['RSI'] < 30 and latest['MACD_Hist'] > 0),  # Oversold with positive momentum
                (latest['Stoch_K'] < 20 and latest['Stoch_D'] < 20),  # Stochastic oversold
                (latest['Price_EMA_5'] > latest['Price_EMA_10'] and latest['Price_EMA_10'] > latest['Price_EMA_20']),  # Upward EMA trend
                (latest['CCI'] < -100),  # Extreme oversold condition
                (latest['ADX'] > 25 and latest['MACD'] > latest['MACD_Signal']),  # Strong trend with bullish MACD
                (latest['ROC'] > 0 and latest['Volume_Ratio'] > 1.2),  # Positive momentum with volume surge
                (latest['MFI'] < 20),  # Low money flow
                (latest['Price_EMA_5'] < latest['BB_Lower'] * 1.01)  # Near lower Bollinger Band
            ]
            
            # Comprehensive sell conditions
            sell_conditions = [
                # Strong bearish indicators
                (latest['RSI'] > 70 and latest['MACD_Hist'] < 0),  # Overbought with negative momentum
                (latest['Stoch_K'] > 80 and latest['Stoch_D'] > 80),  # Stochastic overbought
                (latest['Price_EMA_5'] < latest['Price_EMA_10'] and latest['Price_EMA_10'] < latest['Price_EMA_20']),  # Downward EMA trend
                (latest['CCI'] > 100),  # Extreme overbought condition
                (latest['ADX'] > 25 and latest['MACD'] < latest['MACD_Signal']),  # Strong trend with bearish MACD
                (latest['ROC'] < 0 and latest['Volume_Ratio'] > 1.2),  # Negative momentum with volume surge
                (latest['MFI'] > 80),  # High money flow
                (latest['Price_EMA_5'] > latest['BB_Upper'] * 0.99)  # Near upper Bollinger Band
            ]
            
            # Determine signal based on conditions
            if sum(buy_conditions) >= 3:  # At least 3 buy conditions
                signal = 1
            elif sum(sell_conditions) >= 3:  # At least 3 sell conditions
                signal = -1
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {str(e)}")
            return 0  # Default to hold on error

    def prepare_features(self, data):
        """
        Prepare technical indicators and features
        
        Args:
            data (pd.DataFrame): Raw OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with technical indicators
        """
        try:
            # Create a copy to avoid modifying original data
            df = data.copy()
            
            # Ensure data is sorted by time
            df = df.sort_index()
            
            # Forward fill missing data
            df = df.ffill()
            
            # Engineer features
            df = self._engineer_features(df)
            
            # Select feature columns
            feature_columns = ['Returns', 'Volume', 'Volume_EMA', 
                             'Price_EMA_5', 'Price_EMA_10', 'Price_EMA_20',
                             'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
                             'ATR', 'BB_Upper', 'BB_Lower', 'BB_Width',
                             'Stoch_K', 'Stoch_D', 'ROC', 'MFI',
                             'ADX', 'CCI']
            
            features = df[feature_columns].copy()
            
            # Ensure all features are numeric
            features = features.astype(float)
            
            # Final check for any remaining NaN or inf values
            if features.isnull().any().any() or np.isinf(features.values).any():
                self.logger.warning("Some features still contain NaN or inf values after cleaning")
                features = features.ffill().bfill()  # One final attempt to clean
            
            self.logger.info(f"Prepared {len(features)} valid feature rows")
            self.logger.info(f"Feature columns: {list(features.columns)}")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature preparation error: {str(e)}")
            self.logger.error(f"Error occurred while preparing features: {str(e)}")
            return pd.DataFrame()
            
    def generate_intraday_signals(self, data):
        """
        Generate intraday trading signals based on technical indicators
        
        Args:
            data (pd.DataFrame): DataFrame with technical indicators
            
        Returns:
            int: Trading signal (-1: Sell, 0: Hold, 1: Buy)
        """
        try:
            # Extract latest values
            latest = data.iloc[-1]
            
            # Initialize signal as hold (0)
            signal = 0
            
            # Comprehensive buy conditions
            buy_conditions = [
                # Strong momentum indicators
                (latest['RSI'] < 30 and latest['MACD_Hist'] > 0),  # Oversold with positive momentum
                (latest['Stoch_K'] < 20 and latest['Stoch_D'] < 20),  # Stochastic oversold
                (latest['Price_EMA_5'] > latest['Price_EMA_10'] and latest['Price_EMA_10'] > latest['Price_EMA_20']),  # Upward EMA trend
                (latest['CCI'] < -100),  # Extreme oversold condition
                (latest['ADX'] > 25 and latest['MACD'] > latest['MACD_Signal']),  # Strong trend with bullish MACD
                (latest['ROC'] > 0 and latest['Volume_Ratio'] > 1.2),  # Positive momentum with volume surge
                (latest['MFI'] < 20),  # Low money flow
                (latest['Price_EMA_5'] < latest['BB_Lower'] * 1.01)  # Near lower Bollinger Band
            ]
            
            # Comprehensive sell conditions
            sell_conditions = [
                # Strong bearish indicators
                (latest['RSI'] > 70 and latest['MACD_Hist'] < 0),  # Overbought with negative momentum
                (latest['Stoch_K'] > 80 and latest['Stoch_D'] > 80),  # Stochastic overbought
                (latest['Price_EMA_5'] < latest['Price_EMA_10'] and latest['Price_EMA_10'] < latest['Price_EMA_20']),  # Downward EMA trend
                (latest['CCI'] > 100),  # Extreme overbought condition
                (latest['ADX'] > 25 and latest['MACD'] < latest['MACD_Signal']),  # Strong trend with bearish MACD
                (latest['ROC'] < 0 and latest['Volume_Ratio'] > 1.2),  # Negative momentum with volume surge
                (latest['MFI'] > 80),  # High money flow
                (latest['Price_EMA_5'] > latest['BB_Upper'] * 0.99)  # Near upper Bollinger Band
            ]
            
            # Determine signal based on conditions
            if sum(buy_conditions) >= 3:  # At least 3 buy conditions
                signal = 1
            elif sum(sell_conditions) >= 3:  # At least 3 sell conditions
                signal = -1
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {str(e)}")
            return 0  # Default to hold on error

    def prepare_sequences(self, features):
        """
        Prepare input sequences for LSTM model
        
        Args:
            features (pd.DataFrame): Technical indicator features
            
        Returns:
            tuple: (X, y) sequences and labels
        """
        try:
            if len(features) < self.lookback_period:
                self.logger.error("Insufficient data for sequence preparation")
                return None, None
            
            # Initialize lists for sequences and labels
            X, y = [], []
            
            # Generate sequences
            for i in range(len(features) - self.lookback_period):
                # Get sequence window
                sequence = features.iloc[i:i+self.lookback_period]
                
                # Generate signal for next period
                next_period = features.iloc[i+self.lookback_period:i+self.lookback_period+1]
                if len(next_period) == 0:
                    continue
                    
                next_period_signal = self.generate_intraday_signals(next_period)
                
                # Add to lists
                X.append(sequence.values)
                y.append(next_period_signal)
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            n_samples, n_timesteps, n_features = X.shape
            X_reshaped = X.reshape(n_samples * n_timesteps, n_features)
            X_scaled = self.scaler.fit_transform(X_reshaped)
            X = X_scaled.reshape(n_samples, n_timesteps, n_features)
            
            self.logger.info(f"Prepared sequences: X shape {X.shape}, y shape {y.shape}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Sequence preparation error: {str(e)}")
            return None, None

    def train(self, X, y):
        """
        Train the LSTM model for intraday trading
        
        Args:
            X (np.ndarray): Input sequences
            y (np.ndarray): Target signals
        
        Returns:
            None
        """
        try:
            # Validate input
            if X is None or y is None:
                self.logger.error("Invalid training data")
                return
            
            # One-hot encode labels
            y_categorical = tf.keras.utils.to_categorical(y + 1, num_classes=3)
            
            # Split data into train and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_categorical, test_size=0.2, random_state=42
            )
            
            # Clear previous model if exists
            tf.keras.backend.clear_session()
            
            # Define model architecture
            self.model = tf.keras.Sequential([
                # Input layer
                tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
                
                # LSTM layers with increased complexity
                tf.keras.layers.LSTM(128, return_sequences=True),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(64, return_sequences=True),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.LSTM(32, return_sequences=False),
                tf.keras.layers.Dropout(0.2),
                
                # Dense layers with increased width
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                
                # Output layer
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            
            # Compile model with weighted loss
            class_weights = {0: 1.0, 1: 2.0, 2: 2.0}  # Increase weight for buy/sell signals
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Early stopping with more patience
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=15,  # Increased patience
                restore_best_weights=True,
                mode='min'
            )
            
            # Model checkpointing
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath='best_model.keras', 
                monitor='val_accuracy', 
                save_best_only=True,
                mode='max'
            )
            
            # Train model with class weights
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                class_weight=class_weights,  # Add class weights
                callbacks=[early_stopping, model_checkpoint],
                verbose=1
            )
            
            # Log training results
            self.logger.info("Model training completed")
            self.logger.info(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
            self.logger.info(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            
    def predict(self, X):
        """
        Generate predictions for input sequences
        
        Args:
            X (np.ndarray): Input sequences
        
        Returns:
            np.ndarray: Predicted signals
        """
        try:
            # Validate input
            if X is None or len(X) == 0:
                self.logger.error("Invalid prediction input")
                return np.array([])
            
            # Load best model if not already loaded
            if not hasattr(self, 'model') or self.model is None:
                try:
                    self.model = tf.keras.models.load_model('best_model.keras')
                except Exception as e:
                    self.logger.error(f"No saved model found: {str(e)}")
                    return np.array([])
            
            # Generate predictions
            predictions_prob = self.model.predict(X, verbose=0)
            
            # Convert probabilities to class predictions
            predictions = np.argmax(predictions_prob, axis=1) - 1
            
            return predictions
        
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return np.array([])

    def backtest(self, date=None):
        """
        Perform comprehensive intraday backtesting
        
        Args:
            date (str, optional): Specific date for backtesting
        
        Returns:
            dict: Performance metrics of the trading strategy
        """
        try:
            # Download intraday data
            intraday_data = self.download_intraday_data(date)
            
            if intraday_data.empty:
                self.logger.error("No intraday data available")
                return {}
            
            # Prepare features
            features = self.prepare_features(intraday_data)
            
            if features.empty:
                self.logger.error("Feature preparation failed")
                return {}
            
            # Prepare sequences
            X, y = self.prepare_sequences(features)
            
            if X is None or y is None:
                self.logger.error("Sequence preparation failed")
                return {}
            
            # Train model if not already trained
            if self.model is None:
                self.train(X, y)
            
            # Generate signals
            predictions = self.model.predict(X)
            predicted_signals = np.argmax(predictions, axis=1) - 1
            
            # Create signals DataFrame
            signals_df = pd.DataFrame({
                'Timestamp': features.index[-len(predicted_signals):],
                'Signal': predicted_signals
            })
            
            # Signal distribution
            signal_counts = signals_df['Signal'].value_counts()
            self.logger.info(f"Intraday Signal Distribution:\n{signal_counts}")
            
            # Performance metrics
            performance_metrics = {
                'total_signals': len(signals_df),
                'buy_signals': signal_counts.get(1, 0),
                'sell_signals': signal_counts.get(-1, 0),
                'hold_signals': signal_counts.get(0, 0),
                'date': date or datetime.now().strftime('%Y-%m-%d')
            }
            
            return performance_metrics
        
        except Exception as e:
            self.logger.error(f"Intraday backtesting failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
