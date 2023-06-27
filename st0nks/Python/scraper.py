import pandas as pd
from datetime import datetime, timedelta
from pandas_datareader import data as web
import yfinance as yf
import os

yf.pdr_override()

# Absolute Path to data folder
path = r'../../data/'

# Tickers to get data from
tickers = ["META", "AMZN", "AAPL", "NFLX", "GOOG"]

# Time calcualtion
today = datetime.today()
d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = datetime.today() - timedelta(days=365)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

# Get CSV files from ticker data
for stock in tickers:
    stock_data = web.get_data_yahoo(stock, start=start_date, end=end_date)
    stock_data = stock_data['Adj Close']
    df = pd.DataFrame(stock_data)
    df.to_csv(os.path.join(path, (stock + ".csv")),
              header=False, index=False)
