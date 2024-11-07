# src/scripts/run_backtesting.py

from src.backtesting import Backtester
from src.config import Config

def main():
    backtester = Backtester(data_path=Config.DATA_PROCESSED_PATH)
    backtester.run_backtest()

if __name__ == "__main__":
    main()
