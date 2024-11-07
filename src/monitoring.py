from telegram import Bot
from src.config import Config
from src.utils import setup_logger
import logging

class TelegramNotifier:
    """
    Class to send notifications via Telegram.
    """
    def __init__(self, bot_token, chat_id, log_path='logs/monitoring.log'):
        """
        Initializes the TelegramNotifier.
        
        Parameters:
        - bot_token (str): Telegram bot token.
        - chat_id (str): Telegram chat ID.
        - log_path (str): Path to the log file.
        """
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id
        self.logger = setup_logger('monitoring', log_path)

    def send_message(self, message):
        """
        Sends a message to the specified Telegram chat.
        
        Parameters:
        - message (str): The message to send.
        """
        try:
            self.bot.send_message(chat_id=self.chat_id, text=message)
            self.logger.info(f"Sent message: {message}")
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
