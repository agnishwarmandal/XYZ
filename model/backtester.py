import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, signals_df=None, price_df=None, initial_capital: float = 100000.0):
        """
        Initialize Backtester with signals and price data
        
        Args:
            signals_df (pd.DataFrame): DataFrame with trading signals
            price_df (pd.DataFrame): DataFrame with price information
            initial_capital (float): Starting capital for backtesting
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.signals_df = signals_df
        self.price_df = price_df
        self.positions = []
        self.trades_history = []
        self.equity_curve = [initial_capital]
        self.max_drawdown = 0
        self.win_rate = 0
        self.profit_factor = 0
        self.sharpe_ratio = 0
        self.sortino_ratio = 0
    
    def run_backtest(self) -> Dict:
        """
        Comprehensive backtesting of trading strategy
        
        Returns:
            Dict with performance metrics
        """
        if self.signals_df is None or len(self.signals_df) == 0:
            logger.warning("No signals provided for backtesting")
            return {}
        
        # Reset metrics
        self.current_capital = self.initial_capital
        self.positions = []
        self.trades_history = []
        self.equity_curve = [self.initial_capital]
        
        current_position = 0
        entry_price = 0
        stop_loss = 0
        
        for idx, row in self.signals_df.iterrows():
            current_close = row['Close']
            current_signal = row.get('Signal', 0)
            
            # Entry logic
            if current_position == 0 and current_signal != 0:
                entry_price = current_close
                current_position = current_signal
                stop_loss = row.get('Trailing_Stop', entry_price)
            
            # Exit or stop loss logic
            if current_position != 0:
                if (current_position == 1 and current_close <= stop_loss) or \
                   (current_position == -1 and current_close >= stop_loss) or \
                   current_signal == 0:
                    # Calculate trade performance
                    trade_return = (current_close - entry_price) * current_position
                    self.current_capital += trade_return
                    
                    # Record trade
                    self.trades_history.append({
                        'entry_price': entry_price,
                        'exit_price': current_close,
                        'return': trade_return,
                        'position': current_position
                    })
                    
                    # Reset position
                    current_position = 0
                    entry_price = 0
                    stop_loss = 0
            
            # Update equity curve
            self.equity_curve.append(self.current_capital)
        
        # Calculate performance metrics
        metrics = self._calculate_metrics()
        return metrics
    
    def _calculate_metrics(self) -> Dict:
        """
        Calculate comprehensive trading performance metrics
        
        Returns:
            Dict with performance metrics
        """
        if len(self.equity_curve) < 2:
            return {}
        
        # Total return
        total_return = ((self.equity_curve[-1] - self.initial_capital) / self.initial_capital) * 100
        
        # Win/Loss analysis
        wins = [trade['return'] for trade in self.trades_history if trade['return'] > 0]
        losses = [trade['return'] for trade in self.trades_history if trade['return'] < 0]
        
        win_rate = len(wins) / len(self.trades_history) if self.trades_history else 0
        profit_factor = (sum(wins) / abs(sum(losses))) if losses else 1
        
        # Drawdown calculation
        peak = self.initial_capital
        max_drawdown = 0
        for value in self.equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades_history)
        }
    
    def plot_results(self, save_path='backtest_results.png'):
        """
        Generate and save performance visualization
        
        Args:
            save_path (str): Path to save performance plot
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve, label='Equity Curve')
        plt.title('Trading Strategy Performance')
        plt.xlabel('Trading Days')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
