import pandas as pd
import numpy as np
import tensorflow as tf
from binance import ThreadedWebsocketManager
from src.config import Config
from src.preprocessing import Preprocessor
from src.utils.logger import setup_logger
from src.trading import AutomatedTrader
import time
from concurrent.futures import ThreadPoolExecutor
from src.utils.arg_parser import parse_args

class RealTimePredictor:
    """
    Class to handle real-time data streaming, prediction, and trading.
    """
    def __init__(self, model_path, scaler, time_steps=60, symbol='BTCUSDT'):
        """
        Initializes the RealTimePredictor.
        
        Parameters:
        - model_path (str): Path to the trained model.
        - scaler (StandardScaler): Fitted scaler for feature scaling.
        - time_steps (int): Number of time steps for LSTM input.
        - symbol (str): Trading pair symbol.
        """
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = scaler
        self.time_steps = time_steps
        self.buffer = []
        self.symbol = symbol
        Config.SYMBOL = symbol
        self.features = [
            'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 
            'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'Volume_Change',
            'Bollinger_High', 'Bollinger_Low', 'Bollinger_Width',
            'Stochastic_%K', 'Stochastic_%D',
            'ADX', 'ADX_Pos_Directional', 'ADX_Neg_Directional',
            'Uptrend_Start'
        ]
        self.preprocessor = Preprocessor(
            raw_data_path=Config.DATA_RAW_PATH,
            processed_data_path=Config.DATA_PROCESSED_PATH
        )
        self.trader = AutomatedTrader(
            api_key=Config.BINANCE_API_KEY,
            api_secret=Config.BINANCE_API_SECRET,
            symbol=self.symbol,
            quantity=Config.TRADE_QUANTITY
        )
        self.logger = setup_logger('RealTimePredictor', f'logs/realtime.log')
        self.logger.info(model_path)
    
    def on_message(self, msg):
        self.logger.info("received from stream")
        self.logger.info(msg)
        """
        Callback function to handle incoming WebSocket messages.
        
        Parameters:
        - msg (dict): WebSocket message.
        """
        try:
            # Parse incoming message
            if msg['e'] != 'error':
                # Extract relevant data
                event_time = pd.to_datetime(msg['E'], unit='ms')
                open_price = float(msg['k']['o'])
                high_price = float(msg['k']['h'])
                low_price = float(msg['k']['l'])
                close_price = float(msg['k']['c'])
                volume = float(msg['k']['v'])
                is_kline_closed = msg['k']['x']
                
                self.logger.info(f"is_kline_closed: {is_kline_closed}")
                if is_kline_closed:
                    # Append to buffer
                    self.buffer.append({
                        'Date': event_time,
                        'Open': open_price,
                        'High': high_price,
                        'Low': low_price,
                        'Close': close_price,
                        'Volume': volume,
                        'Uptrend_Start': 0
                    })

                    # Detect if the current kline is the start of an uptrend
                    # For this, we need to check the previous kline's SMA_20 and SMA_50
                    if len(self.buffer) >= 2:
                        prev_close = self.buffer[-2]['Close']
                        # Load a temporary DataFrame to calculate SMAs
                        temp_df = pd.DataFrame(self.buffer[-2:])
                        temp_df.set_index('Date', inplace=True)
                        temp_df = self.preprocessor.handle_missing_values(temp_df)
                        temp_df = self.preprocessor.feature_engineering(temp_df)
                        # Check if the last kline is Uptrend_Start
                        uptrend_start = temp_df['Uptrend_Start'].iloc[-1]
                        if uptrend_start == 1:
                            self.closest_uptrend = {
                                'Date': temp_df.index[-1],
                                'Price': temp_df['Close'].iloc[-1]
                            }
                            self.logger.info(f"Uptrend started for {self.symbol} at {temp_df.index[-1]} with price {temp_df['Close'].iloc[-1]:.2f}")
                            # Optionally, send a Telegram message
                            self.trader.notifier.send_message(
                                f"Uptrend started for {self.symbol} at {temp_df.index[-1]} with price {temp_df['Close'].iloc[-1]:.2f}"
                            )
                    
                    # Calculate the recent lowest price within the current buffer
                    if len(self.buffer) >= self.time_steps:
                        # Create a DataFrame from the buffer
                        buffer_df = pd.DataFrame(self.buffer[-self.time_steps:])
                        buffer_df.set_index('Date', inplace=True)
                        # Identify the recent lowest price and its datetime
                        recent_low_price = buffer_df['Close'].min()
                        recent_low_datetime = buffer_df['Recent_Low_Datetime'].iloc[-1]  # Unix timestamp
                        # Convert Unix timestamp back to datetime for logging
                        recent_low_datetime = pd.to_datetime(recent_low_datetime, unit='s')
                        self.closest_low = {
                            'Date': recent_low_datetime,
                            'Price': recent_low_price
                        }
                        self.logger.info(f"Recent lowest price for {self.symbol}: {recent_low_price:.2f} at {recent_low_datetime}")
                        # Optionally, send a Telegram message
                        self.trader.notifier.send_message(
                            f"Recent lowest price for {self.symbol}: {recent_low_price:.2f} at {recent_low_datetime}"
                        )

                    # Keep only the last 'time_steps' data points
                    if len(self.buffer) > self.time_steps:
                        self.buffer.pop(0)
                    
                    # If buffer is sufficient, make prediction
                    if len(self.buffer) == self.time_steps:
                        df = pd.DataFrame(self.buffer)
                        df.set_index('Date', inplace=True)
                        
                        # Feature engineering
                        df = self.preprocessor.handle_missing_values(df)
                        df = self.preprocessor.feature_engineering(df)
                        
                        # Select features
                        feature_data = df[self.features].tail(1)
                        
                        # Scale features
                        scaled_features = self.scaler.transform(feature_data)
                        
                        # Reshape for LSTM
                        input_sequence = np.expand_dims(scaled_features, axis=0)
                        
                        # Predict
                        prediction_prob = self.model.predict(input_sequence)
                        prediction = (prediction_prob > 0.5).astype(int)[0][0]

                        # Log prediction
                        self.logger.info(f"Reversal predicted for {self.symbol} at {event_time} with probability {prediction_prob[0][0]:.2f}")
                        
                        # Execute trading strategy
                        self.trader.execute_strategy(prediction=prediction, probability=prediction_prob[0][0])

                        # Log closest uptrend details
                        if self.closest_uptrend:
                            uptrend_date = self.closest_uptrend['Date']
                            uptrend_price = self.closest_uptrend['Price']
                            self.logger.info(f"Closest Uptrend Start for {self.symbol}: Date-Time={uptrend_date}, Price={uptrend_price:.2f}")
                            # Optionally, send a Telegram message
                            self.trader.notifier.send_message(
                                f"Closest Uptrend Start for {self.symbol}: Date-Time={uptrend_date}, Price={uptrend_price:.2f}"
                            )
                            # Reset closest_uptrend after logging
                            self.closest_uptrend = None

                        # Log closest recent low details
                        if self.closest_low:
                            low_date = self.closest_low['Date']
                            low_price = self.closest_low['Price']
                            self.logger.info(f"Closest Recent Low for {self.symbol}: Date-Time={low_date}, Price={low_price:.2f}")
                            # Optionally, send a Telegram message
                            self.trader.notifier.send_message(
                                f"Closest Recent Low for {self.symbol}: Date-Time={low_date}, Price={low_price:.2f}"
                            )
                            # Reset closest_low after logging
                            self.closest_low = None
            else:
                self.logger.error(f"Error message received from WebSocket for {self.symbol}: {msg}")
        except Exception as e:
             self.logger.info(e)
    
    def start_stream(self, twm):
        """
        Starts the real-time data stream from Binance.
        
        Parameters:
        - twm (ThreadedWebsocketManager): Binance WebSocket manager instance.
        """
        try:
            # Start Kline Stream
            twm.start_kline_socket(callback=self.on_message, symbol=self.symbol, interval=Config.INTERVAL)
            self.logger.info(f"Started real-time data stream for {self.symbol}...")
            self.logger.info(f"WebSocket connected for {self.symbol}: {twm}")
        except Exception as e:
            self.logger.exception(f"Failed to start stream for {self.symbol}: {e}")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            twm.stop()
            self.logger.info(f"Stopped real-time data stream for {self.symbol}.")

def start_predictor(symbol):
    """
    Initializes and starts the RealTimePredictor for a given symbol.
    
    Parameters:
    - symbol (str): Trading pair symbol.
    """
    raw_path = f'data/raw/{symbol}_{Config.INTERVAL}.csv'
    processed_path = f'data/processed/processed_{symbol}_{Config.INTERVAL}.csv'
    model_path = f'models/lstm_model_{symbol}_{Config.INTERVAL}.h5'
    
    # Update Config paths for the symbol
    Config.SYMBOL = symbol
    Config.DATA_RAW_PATH = raw_path
    Config.DATA_PROCESSED_PATH = processed_path
    Config.MODEL_PATH = model_path
    
    # Initialize Preprocessor and load scaler
    preprocessor = Preprocessor(
        raw_data_path=raw_path,
        processed_data_path=processed_path
    )
    df = preprocessor.load_data()
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.feature_engineering(df)
    features = [
        'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 
        'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'Volume_Change',
        'Bollinger_High', 'Bollinger_Low', 'Bollinger_Width',
        'Stochastic_%K', 'Stochastic_%D',
        'ADX', 'ADX_Pos_Directional', 'ADX_Neg_Directional',
        'Uptrend_Start'
    ]
    df = preprocessor.scale_features(df, features)
    
    predictor = RealTimePredictor(
        model_path=model_path,
        scaler=preprocessor.scaler,
        time_steps=Config.TIME_STEPS,
        symbol=symbol
    )
    
    # Initialize WebSocket Manager
    twm = ThreadedWebsocketManager(api_key=Config.BINANCE_API_KEY, api_secret=Config.BINANCE_API_SECRET)
    twm.start()
    
    # Start real-time prediction in a separate thread
    predictor.start_stream(twm)

# def main():
#     # Handle multiple symbols if provided, separated by commas
#     # symbols = [s.strip() for s in Config.SYMBOL.split(',')] if ',' in Config.SYMBOL else [Config.SYMBOL]

#     args = parse_args()
#     symbols = args.symbol.split(',') if args.symbol else [Config.SYMBOL]

#     print(symbols)
    
#     # Use ThreadPoolExecutor for parallel predictors
#     with ThreadPoolExecutor(max_workers=len(symbols)) as executor:
#         executor.map(start_predictor, symbols)

# if __name__ == "__main__":
#     main()
