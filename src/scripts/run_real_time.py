# src/scripts/run_real_time.py

from src.real_time import RealTimePredictor
from src.config import Config

def main():
    predictor = RealTimePredictor(
        model_path=Config.MODEL_PATH,
        scaler=None,  # Ensure to pass the fitted scaler
        time_steps=Config.TIME_STEPS
    )
    predictor.start_stream()

if __name__ == "__main__":
    main()
