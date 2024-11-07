import argparse
import os

def parse_args():
    """
    Parses command-line arguments.
    
    Returns:
    - argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Cryptocurrency AI Model Runner")
    parser.add_argument('--symbol', type=str, help='Trading pair symbol (e.g., BTCUSDT, ETHUSDT)')
    parser.add_argument('--interval', type=str, default='1h', help='Time interval (e.g., 1h, 1d)')
    parser.add_argument('--lookback', type=str, default='1 year ago UTC', help='Lookback period for historical data')
    # Add other arguments as needed
    
    args = parser.parse_args()
    
    # Override environment variables if arguments are provided
    if args.symbol:
        os.environ['SYMBOL'] = args.symbol
    if args.interval:
        os.environ['INTERVAL'] = args.interval
    if args.lookback:
        os.environ['LOOKBACK'] = args.lookback
    
    return args
