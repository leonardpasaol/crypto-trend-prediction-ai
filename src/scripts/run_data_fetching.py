import sys
import os

from src.utils.arg_parser import parse_args
from src.data_fetcher import DataFetcher
from src.config import Config
from src.utils.logger import setup_logger


logger = setup_logger('data_fetcher_logger', 'logs/data_fetcher.log')

def main():
    # Parse command-line arguments
    args = parse_args()

    # Handle multiple symbols if provided, separated by commas
    # symbols = [s.strip() for s in Config.SYMBOL.split(',')] if ',' in Config.SYMBOL else [Config.SYMBOL]
    symbols = args.symbol.split(',') if args.symbol else [Config.SYMBOL]

    for symbol in symbols:

        raw_path = f'data/raw/{symbol}_{Config.INTERVAL}.csv'
        processed_path = f'data/processed/processed_{symbol}_{Config.INTERVAL}.csv'
        model_path = f'models/lstm_model_{symbol}_{Config.INTERVAL}.h5'

        os.environ['SYMBOL'] = symbol
        # Update Config paths
        Config.SYMBOL = symbol
        Config.DATA_RAW_PATH = raw_path
        Config.DATA_PROCESSED_PATH = processed_path
        Config.MODEL_PATH = model_path
        
        logger.info("Starting the data fetching process.")

        fetcher = DataFetcher(api_key=Config.BINANCE_API_KEY, api_secret=Config.BINANCE_API_SECRET)
        fetcher.set_paths(model_path=model_path, raw_path=raw_path, processed_path=processed_path, symbol=symbol)
        fetcher.run()

if __name__ == "__main__":
    main()
