import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from sklearn.preprocessing import StandardScaler
from src.config import Config

class Preprocessor:
    """
    Class to handle data preprocessing and feature engineering.
    """
    def __init__(self, raw_data_path, processed_data_path):
        """
        Initializes the Preprocessor with file paths.
        
        Parameters:
        - raw_data_path (str): Path to the raw data CSV file.
        - processed_data_path (str): Path to save the processed data CSV file.
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.scaler = StandardScaler()
    
    def load_data(self):
        """
        Loads raw data from the CSV file.
        
        Returns:
        - pd.DataFrame: DataFrame containing raw data.
        """
        df = pd.read_csv(self.raw_data_path, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        return df
    
    def handle_missing_values(self, df):
        """
        Handles missing values by forward-filling.
        
        Parameters:
        - df (pd.DataFrame): DataFrame with potential missing values.
        
        Returns:
        - pd.DataFrame: DataFrame with missing values handled.
        """
        df = df.ffill()
        return df
    
    def feature_engineering(self, df):
        """
        Adds technical indicators to the DataFrame.
        
        Parameters:
        - df (pd.DataFrame): DataFrame to engineer features on.
        
        Returns:
        - pd.DataFrame: DataFrame with new features.
        """
        # Simple Moving Averages
        df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        
        # Exponential Moving Averages
        df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
        df['EMA_50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()
        
        # MACD
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()
        
        # RSI
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        
        # Volume Change
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Bollinger Bands
        bollinger = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['Bollinger_High'] = bollinger.bollinger_hband()
        df['Bollinger_Low'] = bollinger.bollinger_lband()
        df['Bollinger_Width'] = bollinger.bollinger_wband()
        
        # Stochastic Oscillator
        stochastic = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
        df['Stochastic_%K'] = stochastic.stoch()
        df['Stochastic_%D'] = stochastic.stoch_signal()
        
        # ADX
        adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['ADX'] = adx.adx()
        df['ADX_Pos_Directional'] = adx.adx_pos()
        df['ADX_Neg_Directional'] = adx.adx_neg()

        # Detect Uptrend Start using SMA Crossover
        df['Uptrend_Start'] = np.where((df['SMA_20'].shift(1) < df['SMA_50'].shift(1)) & 
                                       (df['SMA_20'] > df['SMA_50']), 1, 0)
        
        # Calculate Recent Lowest Price and Datetime (as Unix Timestamp)
        window_size = 60  # Define the window size (e.g., last 60 candles)
        df['Recent_Low_Price'] = df['Close'].rolling(window=window_size, min_periods=1).min()
        df['Recent_Low_Datetime'] = df['Close'].rolling(window=window_size, min_periods=1).apply(
            lambda x: x.idxmin().timestamp(), raw=False
        )
        
        # Drop NaN values created by indicators
        df.dropna(inplace=True)
        return df
    
    def define_labels(self, df):
        """
        Defines target labels for the model.
        
        Parameters:
        - df (pd.DataFrame): DataFrame to define labels on.
        
        Returns:
        - pd.DataFrame: DataFrame with labels.
        """
        # Define trend direction
        df['Trend'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        
        # Define potential reversal points
        df['Reversal'] = df['Trend'].diff().fillna(0)
        df['Reversal'] = np.where(df['Reversal'] != 0, 1, 0)
        return df
    
    def scale_features(self, df, feature_columns):
        """
        Scales features using StandardScaler.
        
        Parameters:
        - df (pd.DataFrame): DataFrame containing features.
        - feature_columns (list): List of feature column names.
        
        Returns:
        - pd.DataFrame: DataFrame with scaled features.
        """
        df[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        return df
    
    def save_processed_data(self, df):
        """
        Saves the processed data to a CSV file.
        
        Parameters:
        - df (pd.DataFrame): DataFrame to save.
        """
        df.to_csv(self.processed_data_path)
    
    def run(self):
        """
        Executes the entire preprocessing pipeline.
        """
        df = self.load_data()
        df = self.handle_missing_values(df)
        df = self.feature_engineering(df)
        df = self.define_labels(df)
        features = [
            'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 
            'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'Volume_Change',
            'Bollinger_High', 'Bollinger_Low', 'Bollinger_Width',
            'Stochastic_%K', 'Stochastic_%D',
            'ADX', 'ADX_Pos_Directional', 'ADX_Neg_Directional',
            'Uptrend_Start'
        ]

        # Exclude 'Recent_Low_Datetime' from scaling as it's a datetime
        scale_features = [f for f in features if f != 'Uptrend_Start']
        df = self.scale_features(df, scale_features)
        self.save_processed_data(df)
        print(f"Data preprocessed and saved to {self.processed_data_path}")
