from src.utils.arg_parser import parse_args
from src.data_fetcher import DataFetcher
from src.preprocessing import Preprocessor
from src.model import ModelTrainer
from src.config import Config
import os
from concurrent.futures import ThreadPoolExecutor
import logging

from src.utils.logger import setup_logger

logger = setup_logger('full_pipeline', 'logs/pipeline.log')

# logger = logging.getLogger('RunPipeline')
# logger.setLevel(logging.INFO)
# handler = logging.FileHandler('logs/pipeline.log')
# formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
# handler.setFormatter(formatter)

# if not logger.handlers:
#     logger.addHandler(handler)

def process_symbol(symbol):
    logger.info(f"Start parallel processing for {symbol}")
    """
    Executes the full pipeline (fetching, preprocessing, training) for a given symbol.
    
    Parameters:
    - symbol (str): Trading pair symbol.
    """
    # Set the symbol in environment variables
    os.environ['SYMBOL'] = symbol
    # Update Config paths
    Config.DATA_RAW_PATH = f'data/raw/{symbol}_{Config.INTERVAL}.csv'
    Config.DATA_PROCESSED_PATH = f'data/processed/processed_{symbol}_{Config.INTERVAL}.csv'
    Config.MODEL_PATH = f'models/lstm_model_{symbol}.h5'
    
    # Fetch data
    logger.info("Start fetching data ha..")
    fetcher = DataFetcher(api_key=Config.BINANCE_API_KEY, api_secret=Config.BINANCE_API_SECRET)
    fetcher.run()
    
    # Preprocess data
    preprocessor = Preprocessor(
        raw_data_path=Config.DATA_RAW_PATH,
        processed_data_path=Config.DATA_PROCESSED_PATH
    )
    preprocessor.run()
    
    # Train model
    
    trainer = ModelTrainer(
        processed_data_path=Config.DATA_PROCESSED_PATH,
        model_path=Config.MODEL_PATH,
        time_steps=Config.TIME_STEPS,
        imbalance_method='resample'  # or 'class_weight'
    )
    trainer.run()

def main():
    # Parse command-line arguments
    use_parallel_processing = False
    parse_args()
    
    # Handle multiple symbols if provided, separated by commas
    symbols = [s.strip() for s in Config.SYMBOL.split(',')] if ',' in Config.SYMBOL else [Config.SYMBOL]


    if use_parallel_processing == True:
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(process_symbol, symbols)
    else:
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

if __name__ == "__main__":
    main()
