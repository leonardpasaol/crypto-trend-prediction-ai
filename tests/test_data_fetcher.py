import unittest
from src.data_fetcher import DataFetcher
from src.config import Config
import pandas as pd

class TestDataFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = DataFetcher(api_key=Config.BINANCE_API_KEY, api_secret=Config.BINANCE_API_SECRET)
    
    def test_fetch_historical_data(self):
        df = self.fetcher.fetch_historical_data(Config.SYMBOL, Config.INTERVAL, Config.LOOKBACK)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn('Close', df.columns)
    
    def test_save_raw_data(self):
        df = self.fetcher.fetch_historical_data(Config.SYMBOL, Config.INTERVAL, Config.LOOKBACK)
        self.fetcher.save_raw_data(df, Config.DATA_RAW_PATH)
        self.assertTrue(pd.read_csv(Config.DATA_RAW_PATH).shape[0] > 0)

if __name__ == '__main__':
    unittest.main()
