import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.config import Config


def setup_logger_v1(name, log_file, level=logging.INFO):
    """
    Sets up a logger with the specified name and log file.
    
    Parameters:
    - name (str): Name of the logger.
    - log_file (str): File path for the log file.
    - level (int): Logging level.
    
    Returns:
    - logger (logging.Logger): Configured logger.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handler = logging.FileHandler(log_file)        
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger

def setup_logger(name, log_file, level=logging.INFO):
    """
    Sets up a logger with the specified name and log file.
    
    Parameters:
    - name (str): Name of the logger.
    - log_file (str): File path for the log file.
    - level (int): Logging level.
    
    Returns:
    - logger (logging.Logger): Configured logger.
    """
    # Incorporate the symbol into the log file path
    symbol = Config.SYMBOL
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    # Modify log_file to include symbol
    base, ext = os.path.splitext(log_file)
    log_file = f"{base}_{symbol}{ext}"
    
    handler = logging.FileHandler(log_file)        
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger

class PeakPredictor:
    def __init__(self, processed_data_path):
        self.processed_data_path = processed_data_path
    
    def load_processed_data(self):
        df = pd.read_csv(self.processed_data_path, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        return df
    
    def predict_peaks(self, df, window=10):
        # Identify trend direction
        df['Trend'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        df['Uptrend_Start'] = np.where(
            (df['Trend'] == 1) & (df['Trend'].shift(1) == 0), 1, 0
        )
        
        # Calculate rolling maximum for the next 'window' periods
        df['Expected_Peak'] = df['Close'].rolling(window=window, min_periods=1).max().shift(-window)
        
        # Filter uptrend starts with expected peaks
        uptrends = df[df['Uptrend_Start'] == 1]
        uptrends = uptrends.dropna(subset=['Expected_Peak'])
        
        return uptrends[['Close', 'Expected_Peak']]
    
    def visualize_peaks(self, df, uptrends):
        base = Config.SYMBOL.replace('USDT', '')
        plt.figure(figsize=(14,7))
        plt.plot(df.index, df['Close'], label='Close Price')
        plt.plot(df.index, df['Expected_Peak'], label='Expected Peak', linestyle='--')
        plt.scatter(uptrends.index, uptrends['Close'], marker='^', color='r', label='Uptrend Start')
        plt.title(f'{base}/USDT Close Price with Expected Peaks', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price (USDT)', fontsize=14)
        plt.legend()
        plt.show()
    
    def run(self):
        df = self.load_processed_data()
        uptrends = self.predict_peaks(df)
        print(uptrends)
        self.visualize_peaks(df, uptrends)
