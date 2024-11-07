import backtrader as bt
import pandas as pd
from src.config import Config
from src.utils import setup_logger
import logging

class ReversalStrategy(bt.Strategy):
    """
    A simple reversal strategy based on RSI and MACD indicators.
    """
    params = (
        ('printlog', False),
    )

    def log(self, txt, dt=None):
        """
        Logging function for this strategy.
        
        Parameters:
        - txt (str): Text to log.
        - dt (datetime, optional): DateTime of the log.
        """
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            self.logger.info(f'{dt.isoformat()} {txt}')

    def __init__(self):
        """
        Initialize indicators and variables.
        """
        # To keep track of pending orders
        self.order = None
        # Add indicators
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.macd = bt.indicators.MACD(self.data.close)
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=20)

    def notify_order(self, order):
        """
        Notify about order status changes.
        
        Parameters:
        - order: The order object.
        """
        if order.status in [order.Submitted, order.Accepted]:
            return  # No action needed

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}')
            self.order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            self.order = None

    def next(self):
        """
        Define the logic for each new data point.
        """
        self.log(f'Close: {self.data.close[0]:.2f}')
        if self.order:
            return

        # Example strategy:
        # Buy when RSI < 30 and MACD crosses above signal
        if not self.position:
            if self.rsi < 30 and self.macd.macd[0] > self.macd.signal[0]:
                self.log('BUY SIGNAL')
                self.order = self.buy()
        else:
            # Sell when RSI > 70 or MACD crosses below signal
            if self.rsi > 70 or self.macd.macd[0] < self.macd.signal[0]:
                self.log('SELL SIGNAL')
                self.order = self.sell()

class Backtester:
    """
    Class to perform backtesting of trading strategies.
    """
    def __init__(self, data_path, log_path='logs/backtesting.log'):
        """
        Initializes the Backtester.
        
        Parameters:
        - data_path (str): Path to the processed data CSV file.
        - log_path (str): Path to the log file.
        """
        self.data_path = data_path
        self.logger = setup_logger('backtester', log_path)

    def load_data(self):
        """
        Loads historical data from CSV.
        
        Returns:
        - pd.DataFrame: DataFrame containing historical data.
        """
        df = pd.read_csv(self.data_path, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        return df

    def run_backtest(self, strategy_params=None):
        """
        Executes the backtest with the specified strategy.
        
        Parameters:
        - strategy_params (dict, optional): Parameters for the strategy.
        """
        df = self.load_data()
        cerebro = bt.Cerebro()
        cerebro.addstrategy(ReversalStrategy, **(strategy_params or {}))
        
        data = bt.feeds.PandasData(
            dataname=df,
            fromdate=pd.to_datetime('2023-01-01'),
            todate=pd.to_datetime('2024-01-01'),
            reverse=False
        )
        cerebro.adddata(data)
        cerebro.broker.setcash(10000.0)
        cerebro.broker.setcommission(commission=0.001)
        
        self.logger.info('Starting Backtest')
        cerebro.run()
        self.logger.info('Backtest Completed')
        
        final_portfolio = cerebro.broker.getvalue()
        self.logger.info(f'Final Portfolio Value: {final_portfolio:.2f}')
        
        # Plot results
        cerebro.plot(style='candlestick')

def main():
    backtester = Backtester(data_path=Config.DATA_PROCESSED_PATH)
    backtester.run_backtest()

if __name__ == "__main__":
    main()
