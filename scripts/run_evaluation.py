from src.utils.arg_parser import parse_args
from src.evaluation import Evaluator
from src.config import Config

def main():
    # Parse command-line arguments
    parse_args()
    
    # Initialize Evaluator with dynamic paths
    evaluator = Evaluator(
        processed_data_path=Config.DATA_PROCESSED_PATH,
        model_path=Config.MODEL_PATH,
        time_steps=Config.TIME_STEPS
    )
    evaluator.run()

if __name__ == "__main__":
    main()
