from datetime import datetime, timedelta
import ccxt
import pandas as pd

class MexWrapper(object):

    keys = []
    all_exchanges = []
    client = ccxt.bitmex()

    xbit = "BTC/USD"
    eth = "ETH/USD"

    def __init__(self):
        return

    def get_exchange_markets(self):
        return self.client.load_markets()

    def fetch_and_save_historical_bar_data(self, look_back, time_int):
        lb_date = datetime.now() - timedelta(days=look_back)
        hist = mex.client.fetch_ohlcv(self.xbit, timeframe="1d", since=self.client.parse8601(lb_date.strftime('%Y-%m-%dT%H:%M:%S')))
        df = pd.DataFrame(hist)
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        df.to_csv("../hist_data/mex/" + self.xbit.split("/")[0] + ".txt")
        return df
    def read_data_file(self):
        df = pd.DataFrame().from_csv("../hist_data/mex/" + self.xbit.split("/")[0] + ".txt")
        print(df.head())


mex = MexWrapper()
mex.read_data_file()
