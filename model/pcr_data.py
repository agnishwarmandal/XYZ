import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PCRDataFetcher:
    def __init__(self):
        self.nse_url = "https://www.nseindia.com/option-chain"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        self.session = requests.Session()
        self.pcr_history = pd.DataFrame(columns=['timestamp', 'pcr', 'pcr_change'])
        
    def _get_nse_cookies(self):
        """Get cookies from NSE website"""
        try:
            response = self.session.get(self.nse_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                return True
        except Exception as e:
            logger.error(f"Error getting NSE cookies: {str(e)}")
        return False

    def fetch_option_chain_data(self):
        """Fetch option chain data from NSE"""
        if not self._get_nse_cookies():
            return None

        api_url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        try:
            response = self.session.get(api_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching option chain data: {str(e)}")
        return None

    def calculate_pcr(self, option_chain_data):
        """Calculate Put-Call Ratio from option chain data"""
        if not option_chain_data:
            return None

        try:
            total_put_oi = 0
            total_call_oi = 0
            
            for record in option_chain_data['records']['data']:
                if 'PE' in record and 'CE' in record:
                    total_put_oi += record['PE']['openInterest']
                    total_call_oi += record['CE']['openInterest']
            
            if total_call_oi > 0:
                pcr = total_put_oi / total_call_oi
                return round(pcr, 2)
        except Exception as e:
            logger.error(f"Error calculating PCR: {str(e)}")
        return None

    def update_pcr_history(self):
        """Update PCR history with latest data"""
        option_chain_data = self.fetch_option_chain_data()
        if option_chain_data:
            current_pcr = self.calculate_pcr(option_chain_data)
            if current_pcr:
                current_time = datetime.now()
                
                # Calculate PCR change
                if not self.pcr_history.empty:
                    last_pcr = self.pcr_history.iloc[-1]['pcr']
                    pcr_change = current_pcr - last_pcr
                else:
                    pcr_change = 0
                
                # Add new PCR data
                new_data = pd.DataFrame({
                    'timestamp': [current_time],
                    'pcr': [current_pcr],
                    'pcr_change': [pcr_change]
                })
                
                self.pcr_history = pd.concat([self.pcr_history, new_data], ignore_index=True)
                
                # Keep only last 100 records
                if len(self.pcr_history) > 100:
                    self.pcr_history = self.pcr_history.tail(100)
                
                return True
        return False

    def get_pcr_signals(self):
        """Generate trading signals based on PCR and its rate of change"""
        if len(self.pcr_history) < 2:
            return 0  # No signal
            
        current_pcr = self.pcr_history.iloc[-1]['pcr']
        pcr_change = self.pcr_history.iloc[-1]['pcr_change']
        
        # Calculate PCR moving averages
        pcr_ma5 = self.pcr_history['pcr'].tail(5).mean()
        pcr_ma20 = self.pcr_history['pcr'].tail(20).mean() if len(self.pcr_history) >= 20 else pcr_ma5
        
        # Signal generation rules
        signal = 0
        
        # Bullish signals (PCR decreasing rapidly or extremely low)
        if (pcr_change < -0.15 and current_pcr < pcr_ma20) or (current_pcr < 0.7 and pcr_change < -0.1):
            signal = 1  # Buy signal
            
        # Bearish signals (PCR increasing rapidly or extremely high)
        elif (pcr_change > 0.15 and current_pcr > pcr_ma20) or (current_pcr > 1.5 and pcr_change > 0.1):
            signal = -1  # Sell signal
            
        return signal

    def get_latest_pcr_data(self):
        """Get latest PCR data for analysis"""
        if not self.pcr_history.empty:
            return {
                'current_pcr': self.pcr_history.iloc[-1]['pcr'],
                'pcr_change': self.pcr_history.iloc[-1]['pcr_change'],
                'pcr_ma5': self.pcr_history['pcr'].tail(5).mean(),
                'pcr_ma20': self.pcr_history['pcr'].tail(20).mean() if len(self.pcr_history) >= 20 else None
            }
        return None
