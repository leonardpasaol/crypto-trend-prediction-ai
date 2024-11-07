from binance import Client
import pandas as pd
from src.config import Config
from src.utils.logger import setup_logger

class DataFetcher:
    """
    Class to fetch historical cryptocurrency data from Binance.
    """
    def __init__(self, api_key=None, api_secret=None):
        """
        Initializes the Binance client.
        
        Parameters:
        - api_key (str): Binance API key.
        - api_secret (str): Binance API secret.
        """
        self.client = Client(api_key, api_secret)
        self.logger = setup_logger('data_fetcher', 'logs/data_fetcher.log')
    
    def fetch_historical_data(self, symbol, interval, lookback):
        """
        Fetches historical candlestick data from Binance.
        
        Parameters:
        - symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
        - interval (str): Time interval (e.g., '1h', '1d').
        - lookback (str): Timeframe for historical data (e.g., '1 year ago UTC').
        
        Returns:
        - pd.DataFrame: DataFrame containing historical data.
        """
        klines = self.client.get_historical_klines(symbol, interval, lookback)
        df = pd.DataFrame(klines, columns=[
            'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close_time', 'Quote_asset_volume', 'Number_of_trades',
            'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'
        ])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df.set_index('Date', inplace=True)
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df
    
    def save_raw_data(self, df, filepath):
        """
        Saves the fetched raw data to a CSV file.
        
        Parameters:
        - df (pd.DataFrame): DataFrame containing historical data.
        - filepath (str): Path to save the CSV file.
        """
        df.to_csv(filepath)
    
    def run(self):
        """
        Executes the data fetching process and saves the raw data.
        """
        try:
            df = self.fetch_historical_data(
                Config.SYMBOL, Config.INTERVAL, Config.LOOKBACK
            )
            self.save_raw_data(df, Config.DATA_RAW_PATH)
            self.logger.info(f"Data fetched and saved to {Config.DATA_RAW_PATH}")
        except Exception as e:
            self.logger.error(f"Error in data fetching: {e}")
            raise
