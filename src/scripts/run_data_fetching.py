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
    symbols = [s.strip() for s in Config.SYMBOL.split(',')] if ',' in Config.SYMBOL else [Config.SYMBOL]

    print(args)
    for symbol in symbols:
        os.environ['SYMBOL'] = symbol
        logger.info("Starting the data fetching process.")

    # Step 1: Fetch Data
    # Initialize DataFetcher with API keys from Config
    fetcher = DataFetcher(api_key=Config.BINANCE_API_KEY, api_secret=Config.BINANCE_API_SECRET)
    fetcher.run()

if __name__ == "__main__":
    main()
