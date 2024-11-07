# src/scripts/run_pipeline.py

from src.data_fetcher import DataFetcher

from src.preprocessing import Preprocessor
from src.imbalance import ImbalanceHandler
from src.model import ModelTrainer
from src.tuning import HyperparameterTuner
from src.evaluation import Evaluator
from src.utils import PeakPredictor
from src.config import Config
import logging

def main():
    # Setup logging
    logger = logging.getLogger('RunPipeline')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler('logs/pipeline.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    
    logger.info("Starting the data fetching process.")
    # Step 1: Fetch Data
    fetcher = DataFetcher(api_key=Config.BINANCE_API_KEY, api_secret=Config.BINANCE_API_SECRET)
    fetcher.run()

    logger.info("Starting the data preprocessing process.")
    # Step 2: Preprocess Data
    preprocessor = Preprocessor(
        raw_data_path=Config.DATA_RAW_PATH,
        processed_data_path=Config.DATA_PROCESSED_PATH
    )
    preprocessor.run()
    
    logger.info("Starting hyperparameter tuning.")
    # Step 3: Hyperparameter Tuning
    tuner = HyperparameterTuner(processed_data_path=Config.DATA_PROCESSED_PATH, time_steps=Config.TIME_STEPS)
    best_params = tuner.tune(n_trials=50)
    logger.info(f"Best hyperparameters found: {best_params}")
    
    logger.info("Starting model training with best hyperparameters.")
    # Step 4: Train Model with best hyperparameters
    trainer = ModelTrainer(
        processed_data_path=Config.DATA_PROCESSED_PATH,
        model_path=Config.MODEL_PATH,
        time_steps=Config.TIME_STEPS,
        imbalance_method='resample'  # or 'class_weight'
    )
    trainer.run()
    
    logger.info("Starting model evaluation.")
    # Step 5: Evaluate Model
    evaluator = Evaluator(
        processed_data_path=Config.DATA_PROCESSED_PATH,
        model_path=Config.MODEL_PATH,
        time_steps=Config.TIME_STEPS
    )
    evaluator.run()
    
    logger.info("Starting peak prediction.")
    # Step 6: Predict Peaks
    peak_predictor = PeakPredictor(processed_data_path=Config.DATA_PROCESSED_PATH)
    peak_predictor.run()
    
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
