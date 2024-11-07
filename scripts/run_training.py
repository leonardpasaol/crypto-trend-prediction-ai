from src.utils.arg_parser import parse_args
from src.model import ModelTrainer
from src.config import Config

def main():
    # Parse command-line arguments
    parse_args()
    
    # Initialize ModelTrainer with dynamic model path
    trainer = ModelTrainer(
        processed_data_path=Config.DATA_PROCESSED_PATH,
        model_path=Config.MODEL_PATH,
        time_steps=Config.TIME_STEPS,
        imbalance_method='resample'  # or 'class_weight'
    )
    trainer.run()

if __name__ == "__main__":
    main()
