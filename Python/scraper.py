import pandas as pd
from datetime import datetime, timedelta
from pandas_datareader import data as web 
import os

# Absolute Path to data folder
path = r'C:\Users\Reza\Desktop\st0nks\Python\data'

# Tickers to get data from
tickers = ["FB", "AMZN", "AAPL", "NFLX", "GOOG"]

# Time calcualtion
# TODO: Fix so it's hourly and not daily
today = datetime.today()
d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = datetime.today() - timedelta(days=7)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

# Get CSV files from ticker data
for stock in tickers:
 stock_data = web.get_data_yahoo(stock, start = start_date, end = end_date)
 stock_data = stock_data['Adj Close']
 df = pd.DataFrame(stock_data)
 df.to_csv(os.path.join(path, (stock + ".csv")), header=False ,index=False, line_terminator=",")
