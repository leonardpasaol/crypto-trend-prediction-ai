from src.preprocessing import Preprocessor
from src.config import Config
from src.utils.logger import setup_logger

def main():
    # Preprocess data
    preprocessor = Preprocessor(
        raw_data_path=Config.DATA_RAW_PATH,
        processed_data_path=Config.DATA_PROCESSED_PATH
    )
    preprocessor.run()