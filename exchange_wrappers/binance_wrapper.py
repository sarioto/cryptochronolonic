import requests
import json
import pandas as pd


class BinanceUsWrapper(object):

    candlestick_endpoint = "/api/v3/klines?symbol={}&interval={}&limit={}"
    base_endpoint = "https://api.binance.us"
    info_endpoint = "/api/v3/exchangeInfo"
    def __init__(self):
        return

    def get_symbols(self):
        response = requests.get(self.base_endpoint + self.info_endpoint)
        return response.json()["symbols"]

    def get_symbol_hist(self, symbol):
        print("fetching data for: ", symbol)
        resp = requests.get(self.base_endpoint + self.candlestick_endpoint.format(symbol, "1h", 1000))
        return resp.json()
    
    def get_usd_symbols(self):
        syms = self.get_symbols()
        usds = []
        for x in syms:
            if x["symbol"][-3:] == "USD":
                usds.append(x["symbol"])
        return usds

    def store_usd_histories(self):
        syms = self.get_usd_symbols()
        for s in syms:
            data = self.get_symbol_hist(s)
            df = pd.DataFrame(data)
            df = df[df.columns[:6]]
            df.columns = ["date", "open", "high", "low", "close", "volume"]
            df = df.convert_objects(convert_numeric=True)
            df["hl_spread"] = df["high"] - df["low"]
            df["oc_spread"] = df["close"] - df["open"]
            df["rolling_close"] = df["close"].rolling(34).mean() / df["close"]
            #df["rolling_spread"] = df["oc_spread"].rolling(34).mean() / df["oc_spread"]
            df["volume_feature"] = df["volume"].rolling(34).mean() / df["volume"]
            df = df.dropna()
            df.to_csv("../hist_data/binance/" + s + ".txt")


biwrap = BinanceUsWrapper()
biwrap.store_usd_histories()
    