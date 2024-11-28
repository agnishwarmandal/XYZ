import pandas as pd
from datetime import datetime, timedelta
from model.trading_model import NiftyTradingModel
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run backtest and display results
    """
    # Initialize model
    model = NiftyTradingModel()
    
    # Train model if needed
    try:
        model.load_model()
        logger.info("Loaded existing model")
    except:
        logger.info("Training new model...")
        model.train()
        model.save_model()
        logger.info("Model training completed")
    
    # Run backtest for different time periods
    periods = [
        ('1_month', 30),
        ('3_months', 90),
        ('6_months', 180),
        ('1_year', 365)
    ]
    
    results = {}
    
    for period_name, days in periods:
        logger.info(f"Running backtest for {period_name}")
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        metrics = model.backtest(start_date=start_date, end_date=end_date)
        results[period_name] = metrics
        
        logger.info(f"Backtest results for {period_name}:")
        logger.info(f"Total Return: {metrics.get('total_return', 0):.2f}%")
        logger.info(f"Win Rate: {metrics.get('win_rate', 0) * 100:.2f}%")
        logger.info(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        logger.info(f"Total Trades: {metrics.get('total_trades', 0)}")
        logger.info("-" * 50)
    
    # Save results to file
    with open('backtest_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info("Backtest results saved to backtest_results.json")

if __name__ == "__main__":
    main()
