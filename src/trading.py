from binance.client import Client
from binance.enums import *
from src.config import Config
from src.utils.logger import setup_logger
from src.monitoring import TelegramNotifier

class AutomatedTrader:
    """
    Class to handle automated trading based on model predictions.
    """
    def __init__(self, api_key, api_secret, symbol, quantity):
        """
        Initializes the AutomatedTrader.
        
        Parameters:
        - api_key (str): Binance API key.
        - api_secret (str): Binance API secret.
        - symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
        - quantity (float): Quantity to trade.
        """
        self.client = Client(api_key, api_secret)
        self.symbol = symbol
        self.quantity = quantity
        self.logger = setup_logger('AutomatedTrader', 'logs/trading.log')
        self.notifier = TelegramNotifier(
            bot_token=Config.TELEGRAM_BOT_TOKEN,
            chat_id=Config.TELEGRAM_CHAT_ID,
            log_path=f'logs/monitoring_{symbol}.log'
        )
    
    def place_order(self, side, order_type=ORDER_TYPE_MARKET):
        """
        Places a buy or sell order on Binance.
        
        Parameters:
        - side (str): 'BUY' or 'SELL'.
        - order_type (str): Type of order (default: MARKET).
        
        Returns:
        - dict or None: Order details if successful, None otherwise.
        """
        try:
            order = self.client.create_order(
                symbol=self.symbol,
                side=side,
                type=order_type,
                quantity=self.quantity
            )
            self.logger.info(f"Order placed: {side} {self.quantity} {self.symbol}")
            self.notifier.send_message(f"Order placed: {side} {self.quantity} {self.symbol} at price {order['fills'][0]['price']}")
            return order
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            self.notifier.send_message(f"Error placing order: {e}")
            return None
    
    def execute_strategy(self, prediction, probability):
        """
        Executes trading strategy based on prediction.
        
        Parameters:
        - prediction (int): 1 for reversal, 0 for downtrend.
        - probability (float): Probability of the prediction.
        """
        threshold = 0.7  # Example threshold
        if prediction == 1 and probability >= threshold:
            # Example: Buy order
            self.logger.info(f"Reversal detected with probability {probability:.2f}. Executing BUY order.")
            self.notifier.send_message(f"Reversal detected with probability {probability:.2f}. Executing BUY order.")
            self.place_order(SIDE_BUY)
        elif prediction == 0 and probability >= threshold:
            # Example: Sell order
            self.logger.info(f"Downtrend detected with probability {probability:.2f}. Executing SELL order.")
            self.notifier.send_message(f"Downtrend detected with probability {probability:.2f}. Executing SELL order.")
            self.place_order(SIDE_SELL)
        else:
            self.logger.info(f"No significant reversal detected. No action taken.")
            self.notifier.send_message(f"No significant reversal detected. No action taken.")
