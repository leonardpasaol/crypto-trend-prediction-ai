import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath('../src'))

# Load environment variables from .env file
load_dotenv()

class Config:
    # Binance API configuration
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
    
    # Telegram configurations
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # Data parameters
    SYMBOL = os.getenv('SYMBOL', 'BTCUSDT')  # Default symbol
    INTERVAL = '5m' # reference: https://developers.binance.com/docs/derivatives/coin-margined-futures/market-data/Kline-Candlestick-Data
    LOOKBACK = '12 hours ago UTC+8'
    
    # File paths
    DATA_RAW_PATH = f'data/raw/{SYMBOL}_{INTERVAL}.csv'
    DATA_PROCESSED_PATH = f'data/processed/processed_{SYMBOL}_{INTERVAL}.csv'
    
    # Model parameters
    TIME_STEPS = 60
    EPOCHS = 20
    BATCH_SIZE = 64
    MODEL_PATH = f'models/lstm_model_{SYMBOL}.h5'
    
    # Trading configurations
    TRADE_QUANTITY = 0.001  # Example: 0.001 BTC per trade
    
    # Other configurations
    RANDOM_STATE = 42
