import time

from src.utils.arg_parser import parse_args
from src.evaluation import Evaluator
from src.config import Config

def main():
    # Parse command-line arguments
    args = parse_args()
    symbols = args.symbol.split(',') if args.symbol else [Config.SYMBOL]

    for symbol in symbols:
        processed_path = f'data/processed/processed_{symbol}_{Config.INTERVAL}.csv'
        model_path = f'models/lstm_model_{symbol}_{Config.INTERVAL}.h5'
        Config.SYMBOL = symbol
        # Initialize Evaluator with dynamic paths
        evaluator = Evaluator(
            processed_data_path=processed_path,
            model_path=model_path,
            time_steps=Config.TIME_STEPS
        )
        evaluator.run()

        time.sleep(60)

if __name__ == "__main__":
    main()
