from src.utils.peak_predictor import PeakPredictor
from src.config import Config
from src.utils.logger import setup_logger
from src.utils.arg_parser import parse_args

def main():
    args = parse_args()

    # Handle multiple symbols if provided, separated by commas
    symbols = args.symbol.split(',') if args.symbol else [Config.SYMBOL]

    logger = setup_logger("peak_predictor", "logs/peak_predictor.log")

    for symbol in symbols:
        Config.SYMBOL = symbol
        processed_path = f'data/processed/processed_{symbol}_{Config.INTERVAL}.csv'
        
        try:
            logger.info(f"Starting peak prediction for {symbol}.")
            peak_predictor = PeakPredictor(processed_data_path=processed_path)
            peak_predictor.run()
            logger.info("Peak predictor completed successfully.")
        except Exception as e:
            logger.info(e)
        
        

if __name__ == "__main__":
    main()
