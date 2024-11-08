import os
import time

from concurrent.futures import ThreadPoolExecutor

from src.evaluation import Evaluator
from src.tuning import HyperparameterTuner
from src.data_fetcher import DataFetcher
from src.preprocessing import Preprocessor
from src.model import ModelTrainer
from src.config import Config

from src.utils.logger import setup_logger
from src.utils.peak_predictor import PeakPredictor
from src.utils.arg_parser import parse_args

def process_symbol(symbol):
    logger = setup_logger('full_pipeline', 'logs/pipeline.log')
    logger.info(f"Start parallel processing for {symbol}")

    raw_path = f'data/raw/{symbol}_{Config.INTERVAL}.csv'
    processed_path = f'data/processed/processed_{symbol}_{Config.INTERVAL}.csv'
    model_path = f'models/lstm_model_{symbol}_{Config.INTERVAL}.h5'

    # Set the symbol in environment variables
    os.environ['SYMBOL'] = symbol
    Config.SYMBOL = symbol
    # Update Config paths
    Config.DATA_RAW_PATH = raw_path
    Config.DATA_PROCESSED_PATH = processed_path
    Config.MODEL_PATH = model_path
    
    # Step 1: Fetch Data
    logger.info(f"{symbol}_{Config.INTERVAL}: Start fetching data..")
    fetcher = DataFetcher(api_key=Config.BINANCE_API_KEY, api_secret=Config.BINANCE_API_SECRET)
    fetcher.set_paths(model_path=model_path, raw_path=raw_path, processed_path=processed_path, symbol=symbol)
    fetcher.run()
    
    # Step 2: Preprocess Data
    logger.info(f"{symbol}_{Config.INTERVAL} Starting preprocessor..")
    logger.info(f"raw path: {raw_path}")
    logger.info(f"processed_path: {processed_path}")
    preprocessor = Preprocessor(
        raw_data_path=raw_path,
        processed_data_path=processed_path
    )
    preprocessor.run()

    # Step 3: Hyperparameter Tuning
    logger.info(f"{symbol}_{Config.INTERVAL} Starting hyperparameter tuning.")
    tuner = HyperparameterTuner(processed_data_path=processed_path, time_steps=Config.TIME_STEPS)
    best_params = tuner.tune(n_trials=50)
    logger.info(f"{symbol}_{Config.INTERVAL} Best hyperparameters found: {best_params}")
    
    # Step 4: Train Model with best hyperparameters
    logger.info(f"{symbol}_{Config.INTERVAL} Starting model training with best hyperparameters.")
    trainer = ModelTrainer(
        processed_data_path=processed_path,
        model_path=model_path,
        time_steps=Config.TIME_STEPS,
        imbalance_method='resample'  # or 'class_weight'
    )
    trainer.run()

    # Step 5: Evaluate Model
    evaluator = Evaluator(
        processed_data_path=processed_path,
        model_path=model_path,
        time_steps=Config.TIME_STEPS
    )
    evaluator.run()

    logger.info("Starting peak prediction.")
    # Step 6: Predict Peaks
    peak_predictor = PeakPredictor(processed_data_path=processed_path)
    peak_predictor.run()

    logger.info(f"{symbol}_{Config.INTERVAL} Done full pipeline.")

def main():
    # Parse command-line arguments
    args = parse_args()

    # Handle multiple symbols if provided, separated by commas
    symbols = args.symbol.split(',') if args.symbol else [Config.SYMBOL]
    threaded = args.thread

    logger = setup_logger('full_pipeline', 'logs/pipeline.log')
    
    # # Handle multiple symbols if provided, separated by commas
    # symbols = [s.strip() for s in Config.SYMBOL.split(',')] if ',' in Config.SYMBOL else [Config.SYMBOL]
    print(f"threaded: {threaded}")
    if threaded == True:
        logger.info(f"running full pipeline in thread mode")
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(process_symbol, symbols)
    else:
        logger.info(f"running full pipeline in loop mode")
        for symbol in symbols:
            process_symbol(symbol)

            time.sleep(60)

if __name__ == "__main__":
    main()
