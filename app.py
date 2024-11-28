from flask import Flask, render_template, jsonify
from model.trading_model import NiftyTradingModel
import pandas as pd
import threading
import time
from datetime import datetime, timedelta
import yfinance as yf

app = Flask(__name__)
model = NiftyTradingModel()

# Global variables for storing signals and performance
current_signals = None
historical_performance = None
is_market_open = False

def initialize_model():
    global model, historical_performance
    # Download and prepare data
    data = model.download_data()
    
    # Build and train model
    X, y = model.prepare_sequences(data)
    model.build_model()
    model.train(X, y)
    
    # Generate historical signals
    historical_performance = model.generate_signals(data)
    
def update_live_signals():
    global current_signals, is_market_open
    while True:
        now = datetime.now()
        # Check if market is open (9:15 AM to 3:30 PM IST, Monday to Friday)
        is_market_open = (
            now.weekday() < 5 and  # Monday to Friday
            datetime.strptime('09:15:00', '%H:%M:%S').time() <= now.time() <= datetime.strptime('15:30:00', '%H:%M:%S').time()
        )
        
        if is_market_open:
            # Get latest data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)  # Get more data than needed for feature calculation
            nifty_data = yf.download('^NSEI', start=start_date, end=end_date, interval='5m')
            
            if not nifty_data.empty:
                prepared_data = model._prepare_data(nifty_data)
                current_signals = model.generate_signals(prepared_data)
        
        time.sleep(300)  # Update every 5 minutes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/historical_performance')
def get_historical_performance():
    if historical_performance is not None:
        return jsonify({
            'dates': historical_performance.index.strftime('%Y-%m-%d').tolist(),
            'signals': historical_performance['Signal'].tolist(),
            'returns': historical_performance['Returns'].tolist(),
            'prices': historical_performance['Close'].tolist(),
            'trailing_stops': historical_performance['Trailing_Stop'].tolist()
        })
    return jsonify({'error': 'Historical performance not available'})

@app.route('/api/current_signal')
def get_current_signal():
    if current_signals is not None and not current_signals.empty:
        latest_signal = current_signals.iloc[-1]
        
        # Get latest PCR data
        pcr_data = model.pcr_fetcher.get_latest_pcr_data()
        
        return jsonify({
            'timestamp': latest_signal.name.strftime('%Y-%m-%d %H:%M:%S'),
            'signal': int(latest_signal['Signal']),
            'price': float(latest_signal['Close']),
            'trailing_stop': float(latest_signal['Trailing_Stop']),
            'is_market_open': is_market_open,
            'pcr_data': pcr_data
        })
    return jsonify({
        'signal': 0,
        'price': None,
        'trailing_stop': None,
        'is_market_open': is_market_open,
        'pcr_data': None
    })

if __name__ == '__main__':
    # Initialize model in a separate thread
    init_thread = threading.Thread(target=initialize_model)
    init_thread.start()
    
    # Start live signal updates in a separate thread
    signal_thread = threading.Thread(target=update_live_signals)
    signal_thread.daemon = True
    signal_thread.start()
    
    # Run Flask app
    app.run(debug=True, use_reloader=False)
