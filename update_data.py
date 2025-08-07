import yfinance as yf
import os
from datetime import datetime

STOCKS = ['NVDA', 'MSFT', 'GOOGL', 'META', 'AMZN']
START_DATE = '2020-01-01'
DATA_DIR = 'data'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def update_all():
    today = datetime.today().strftime('%Y-%m-%d')
    for ticker in STOCKS:
        df = yf.download(ticker, start=START_DATE, end=today, progress=False)
        if not df.empty:
            df.to_csv(os.path.join(DATA_DIR, f"{ticker}.csv"))

if __name__ == "__main__":
    update_all()