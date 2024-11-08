import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

        print(uptrends[['Trend', 'Uptrend_Start', 'Close', 'Expected_Peak']])
        
        return uptrends[['Close', 'Expected_Peak']]
    
    def visualize_peaks(self, df, uptrends):
        plt.figure(figsize=(14,7))
        plt.plot(df.index, df['Close'], label='Close Price')
        plt.plot(df.index, df['Expected_Peak'], label='Expected Peak', linestyle='--')
        plt.scatter(uptrends.index, uptrends['Close'], marker='^', color='r', label='Uptrend Start')
        plt.title('BTC/USDT Close Price with Expected Peaks', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price (USDT)', fontsize=14)
        plt.legend()
        plt.show()
    
    def run(self):
        df = self.load_processed_data()
        uptrends = self.predict_peaks(df)
        print(uptrends)
        self.visualize_peaks(df, uptrends)
