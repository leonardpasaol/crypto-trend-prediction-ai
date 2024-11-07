import sys
from src.utils.arg_parser import parse_args
from src.data_fetcher import DataFetcher
from src.config import Config
import os
from concurrent.futures import ThreadPoolExecutor
import time

def fetch_and_save(symbol):
    """
    Fetches and saves data for a given symbol.
    
    Parameters:
    - symbol (str): Trading pair symbol.
    """
    # Set the symbol in environment variables
    os.environ['SYMBOL'] = symbol
    # Update Config paths
    Config.DATA_RAW_PATH = f'data/raw/{symbol}_{Config.INTERVAL}.csv'
    Config.DATA_PROCESSED_PATH = f'data/processed/processed_{symbol}_{Config.INTERVAL}.csv'
    Config.MODEL_PATH = f'models/lstm_model_{symbol}.h5'
    
    # Initialize DataFetcher
    fetcher = DataFetcher(api_key=Config.BINANCE_API_KEY, api_secret=Config.BINANCE_API_SECRET)
    fetcher.run()
    

def main_v1():
    # Parse command-line arguments
    args = parse_args()
    
    # Handle multiple symbols if provided
    symbols = [args.symbol] if args.symbol else [Config.SYMBOL]
    
    for symbol in symbols:
        # Update environment variable temporarily
        os.environ['SYMBOL'] = symbol
        # Update Config paths based on the new symbol
        Config.DATA_RAW_PATH = f'data/raw/{symbol}_{Config.INTERVAL}.csv'
        Config.DATA_PROCESSED_PATH = f'data/processed/processed_{symbol}_{Config.INTERVAL}.csv'
        Config.MODEL_PATH = f'models/lstm_model_{symbol}.h5'
        
        # Initialize DataFetcher
        fetcher = DataFetcher(api_key=Config.BINANCE_API_KEY, api_secret=Config.BINANCE_API_SECRET)
        fetcher.run()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Handle multiple symbols if provided, separated by commas
    symbols = args.symbol.split(',') if args.symbol else [Config.SYMBOL]

    print(symbols)
    
    for symbol in symbols:
        print(f"fetching {symbol}")

        os.environ['SYMBOL'] = symbol
        # Update Config paths
        Config.DATA_RAW_PATH = f'data/raw/{symbol}_{Config.INTERVAL}.csv'
        Config.DATA_PROCESSED_PATH = f'data/processed/processed_{symbol}_{Config.INTERVAL}.csv'
        Config.MODEL_PATH = f'models/lstm_model_{symbol}.h5'
        
        fetcher = DataFetcher(api_key=Config.BINANCE_API_KEY, api_secret=Config.BINANCE_API_SECRET)
        fetcher.run()

        time.sleep(60)
    
    # # Use ThreadPoolExecutor for parallel fetching
    # with ThreadPoolExecutor(max_workers=5) as executor:
    #     executor.map(fetch_and_save, symbols)

if __name__ == "__main__":
    main()
