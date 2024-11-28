# Nifty 50 Trading Algorithm

A sophisticated deep learning-based trading algorithm for the Nifty 50 Index that generates automated trading signals with risk management.

## Features

- Deep Learning model for predicting market movements
- Automated trading signals for long (Call) and short (Put) positions
- Real-time market monitoring during trading hours
- Trailing stop-loss implementation (0.5% maximum loss per trade)
- Interactive web dashboard for monitoring performance
- Historical trade analysis and performance metrics

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

## System Components

### 1. Trading Model (`model/trading_model.py`)
- LSTM-based deep learning model
- Technical indicator generation
- Signal generation with risk management
- Historical performance tracking

### 2. Web Application (`app.py`)
- Flask-based web server
- Real-time data updates
- API endpoints for frontend
- Market hours monitoring

### 3. Frontend (`templates/` and `static/`)
- Interactive dashboard
- Real-time signal updates
- Performance visualization
- Historical trade table

## Risk Management

- Maximum loss per trade: 0.5%
- Trailing stop-loss implementation
- Market hours validation (9:15 AM to 3:30 PM IST, Monday to Friday)

## Data Sources

- Historical data: Yahoo Finance
- Real-time data: 5-minute intervals from Yahoo Finance

## Technical Indicators Used

- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands

## Model Architecture

- LSTM layers for sequence learning
- Dropout layers for regularization
- Dense layers for final prediction
- Softmax activation for signal classification

## Performance Monitoring

The dashboard provides:
- Live trading signals
- Current market status
- Performance metrics visualization
- Historical trade log with returns

## Disclaimer

This trading algorithm is for educational and research purposes only. Always conduct your own research and risk assessment before making any investment decisions.
