# src/scripts/run_real_time.py
import os

from binance import ThreadedWebsocketManager
from concurrent.futures import ThreadPoolExecutor

from src.preprocessing import Preprocessor
from src.real_time import RealTimePredictor
from src.config import Config
from src.utils.arg_parser import parse_args

from src.utils.logger import setup_logger



def start_predictor(symbol):
    Config.SYMBOL = symbol
    logger =setup_logger('RunRealTimeLogger', 'logs/realtime.log')
    logger.info(f"start_predictor for {symbol}")
    """
    Initializes and starts the RealTimePredictor for a given symbol.
    
    Parameters:
    - symbol (str): Trading pair symbol.
    """
    raw_path = f'data/raw/{symbol}_{Config.INTERVAL}.csv'
    processed_path = f'data/processed/processed_{symbol}_{Config.INTERVAL}.csv'
    model_path = f'models/lstm_model_{symbol}_{Config.INTERVAL}.h5'

    # Update Config paths for the symbol
    Config.SYMBOL = symbol
    Config.DATA_RAW_PATH = raw_path
    Config.DATA_PROCESSED_PATH = processed_path
    Config.MODEL_PATH = model_path
    
    # Initialize Preprocessor and load scaler
    logger.info("starting preprocessor")
    logger.info(raw_path)
    logger.info(processed_path)
    try:
        preprocessor = Preprocessor(
            raw_data_path=raw_path,
            processed_data_path=processed_path
        )
        df = preprocessor.load_data()
        df = preprocessor.handle_missing_values(df)
        df = preprocessor.feature_engineering(df)
        features = [
            'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 
            'MACD', 'MACD_Signal', 'MACD_Diff', 'RSI', 'Volume_Change',
            'Bollinger_High', 'Bollinger_Low', 'Bollinger_Width',
            'Stochastic_%K', 'Stochastic_%D',
            'ADX', 'ADX_Pos_Directional', 'ADX_Neg_Directional'
        ]
        df = preprocessor.scale_features(df, features)
    except Exception as e:
        logger.info(e)

    
    try:
        logger.info("starting predictor")
        predictor = RealTimePredictor(
            model_path=model_path,
            scaler=preprocessor.scaler,
            time_steps=Config.TIME_STEPS,
            symbol=symbol
        )
    except Exception as e:
        logger.info(e)
    
    logger.info("Initialize WebSocket Manager")
    # Initialize WebSocket Manager
    twm = ThreadedWebsocketManager(api_key=Config.BINANCE_API_KEY, api_secret=Config.BINANCE_API_SECRET)
    twm.start()
    
    # Start real-time prediction in a separate thread
    logger.info("starting stream from twm")
    predictor.start_stream(twm)

def main():
    args = parse_args()

    # Handle multiple symbols if provided, separated by commas
    # symbols = [s.strip() for s in Config.SYMBOL.split(',')] if ',' in Config.SYMBOL else [Config.SYMBOL]
    symbols = args.symbol.split(',') if args.symbol else [Config.SYMBOL]
    print(symbols)
    
    # Use ThreadPoolExecutor for parallel predictors
    with ThreadPoolExecutor(max_workers=len(symbols)) as executor:
        executor.map(start_predictor, symbols)

def main_2():
    args = parse_args()

    # Handle multiple symbols if provided, separated by commas
    symbols = args.symbol.split(',') if args.symbol else [Config.SYMBOL]
    print(symbols)

    if len(symbols) > 1:
        print("too many too handle")
        return

    symbol = symbols[0]
    os.environ['SYMBOL'] = symbol
    Config.SYMBOL = symbol

    model_path = f'models/lstm_model_{symbol}_{Config.INTERVAL}.h5'
    
    predictor = RealTimePredictor(
        model_path=model_path,
        scaler=None,  # Ensure to pass the fitted scaler
        time_steps=Config.TIME_STEPS,
        symbol=symbol
    )
    predictor.start_stream()

if __name__ == "__main__":
    main()
