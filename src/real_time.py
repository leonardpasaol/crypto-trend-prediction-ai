import pandas as pd
import numpy as np
import tensorflow as tf
from binance import ThreadedWebsocketManager
from src.config import Config
from src.preprocessing import Preprocessor
from src.utils import setup_logger
from src.trading import AutomatedTrader
import time
from concurrent.futures import ThreadPoolExecutor

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
        self.features = [
            'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 
            'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'Volume_Change',
            'Bollinger_High', 'Bollinger_Low', 'Bollinger_Width',
            'Stochastic_%K', 'Stochastic_%D',
            'ADX', 'ADX_Pos_Directional', 'ADX_Neg_Directional'
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
        self.logger = setup_logger('RealTimePredictor', f'logs/pipeline_{self.symbol}.log')
    
    def on_message(self, msg):
        """
        Callback function to handle incoming WebSocket messages.
        
        Parameters:
        - msg (dict): WebSocket message.
        """
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
            
            if is_kline_closed:
                # Append to buffer
                self.buffer.append({
                    'Date': event_time,
                    'Open': open_price,
                    'High': high_price,
                    'Low': low_price,
                    'Close': close_price,
                    'Volume': volume
                })
                
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
        else:
            self.logger.error(f"Error message received from WebSocket for {self.symbol}: {msg}")
    
    def start_stream(self, twm):
        """
        Starts the real-time data stream from Binance.
        
        Parameters:
        - twm (ThreadedWebsocketManager): Binance WebSocket manager instance.
        """
        # Start Kline Stream
        twm.start_kline_socket(callback=self.on_message, symbol=self.symbol, interval=Config.INTERVAL)
        
        self.logger.info(f"Started real-time data stream for {self.symbol}...")
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
    # Update Config paths for the symbol
    Config.DATA_RAW_PATH = f'data/raw/{symbol}_{Config.INTERVAL}.csv'
    Config.DATA_PROCESSED_PATH = f'data/processed/processed_{symbol}_{Config.INTERVAL}.csv'
    Config.MODEL_PATH = f'models/lstm_model_{symbol}.h5'
    
    # Initialize Preprocessor and load scaler
    preprocessor = Preprocessor(
        raw_data_path=Config.DATA_RAW_PATH,
        processed_data_path=Config.DATA_PROCESSED_PATH
    )
    df = preprocessor.load_data()
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.feature_engineering(df)
    features = [
        'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 
        'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'Volume_Change',
        'Bollinger_High', 'Bollinger_Low', 'Bollinger_Width',
        'Stochastic_%K', 'Stochastic_%D',
        'ADX', 'ADX_Pos_Directional', 'ADX_Neg_Directional'
    ]
    df = preprocessor.scale_features(df, features)
    
    predictor = RealTimePredictor(
        model_path=Config.MODEL_PATH,
        scaler=preprocessor.scaler,
        time_steps=Config.TIME_STEPS,
        symbol=symbol
    )
    
    # Initialize WebSocket Manager
    twm = ThreadedWebsocketManager(api_key=Config.BINANCE_API_KEY, api_secret=Config.BINANCE_API_SECRET)
    twm.start()
    
    # Start real-time prediction in a separate thread
    predictor.start_stream(twm)

def main():
    # Parse command-line arguments
    parse_args()
    
    # Handle multiple symbols if provided, separated by commas
    symbols = [s.strip() for s in Config.SYMBOL.split(',')] if ',' in Config.SYMBOL else [Config.SYMBOL]
    
    # Use ThreadPoolExecutor for parallel predictors
    with ThreadPoolExecutor(max_workers=len(symbols)) as executor:
        executor.map(start_predictor, symbols)

if __name__ == "__main__":
    main()
