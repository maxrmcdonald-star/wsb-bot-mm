import time # <-- THIS IS THE FIX
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime
import config # Your configuration file

# --- List of tickers to test from your last output ---
tickers_to_test = ["NVDA", "KSS", "TSLA", "PM", "AMZN", "ALL", "PANW", "CC", "OKTA", "MDB"]

# --- API Keys from your config file ---
ALPHA_VANTAGE_API_KEY = config.ALPHA_VANTAGE_API_KEY
FINNHUB_API_KEY = config.FINNHUB_API_KEY

print("="*60)
print("     RUNNING EARNINGS API TROUBLESHOOTING SCRIPT")
print("="*60)

# --- Function 1: Test yfinance ---
def get_earnings_yfinance(ticker_str):
    try:
        ticker = yf.Ticker(ticker_str)
        dates = ticker.get_earnings_dates(limit=4) # Look at last 4 quarters
        if dates is None or dates.empty:
            return "No data"
        
        today = pd.to_datetime(datetime.now().date())
        future_dates = dates[dates.index >= today]
        
        if not future_dates.empty:
            return future_dates.index[0].strftime('%Y-%m-%d')
        return "No upcoming date"
    except Exception:
        return "Error"

# --- Function 2: Test Alpha Vantage (Direct API Call) ---
def get_earnings_alpha_vantage(ticker_str):
    try:
        url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker_str}&apikey={ALPHA_VANTAGE_API_KEY}'
        r = requests.get(url)
        data = r.json()
        
        if 'quarterlyEarnings' in data and data['quarterlyEarnings']:
            # The dates are often for the future, find the first one with an estimated date
            for report in data['quarterlyEarnings']:
                if 'estimatedDate' in report and report['estimatedDate'] != 'None':
                    return report['estimatedDate']
            return "No upcoming date"
        return "No data"
    except Exception:
        return "Error"

# --- Function 3: Test Finnhub ---
def get_earnings_finnhub(ticker_str):
    try:
        url = f'https://finnhub.io/api/v1/calendar/earnings?symbol={ticker_str}&token={FINNHUB_API_KEY}'
        r = requests.get(url)
        data = r.json()
        
        if 'earningsCalendar' in data and data['earningsCalendar']:
            # Find the closest upcoming date
            today = datetime.now().date()
            future_dates = [
                datetime.strptime(item['date'], '%Y-%m-%d').date() 
                for item in data['earningsCalendar'] 
                if datetime.strptime(item['date'], '%Y-%m-%d').date() >= today
            ]
            if future_dates:
                return min(future_dates).strftime('%Y-%m-%d')
            return "No upcoming date"
        return "No data"
    except Exception:
        return "Error"

# --- Main testing loop ---
results = []
for i, ticker in enumerate(tickers_to_test):
    print(f"Testing Ticker: {ticker} ({i+1}/{len(tickers_to_test)})...")
    
    yfinance_result = get_earnings_yfinance(ticker)
    alpha_vantage_result = get_earnings_alpha_vantage(ticker)
    finnhub_result = get_earnings_finnhub(ticker)
    
    results.append({
        "Ticker": ticker,
        "yfinance": yfinance_result,
        "Alpha Vantage": alpha_vantage_result,
        "Finnhub": finnhub_result
    })
    # Alpha Vantage has a strict 5 calls/minute limit, so we must pause.
    time.sleep(15) 

# --- Print the final comparison table ---
results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("               API COMPARISON RESULTS")
print("="*60)
print(results_df)
print("\nThis table shows the NEXT upcoming earnings date found by each API.")
