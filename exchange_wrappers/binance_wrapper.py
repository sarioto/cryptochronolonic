import requests
import json
import pandas as pd


class BinanceUsWrapper(object):

    candlestick_endpoint = "/api/v3/klines?symbol={}&interval={}"
    base_endpoint = "https://api.binance.us"
    info_endpoint = "/api/v3/exchangeInfo"
    def __init__(self):
        return

    def get_symbols(self):
        response = requests.get(self.base_endpoint + self.info_endpoint)
        return response.json()["symbols"]

    def get_symbol_hist(self):
        symbol = self.get_symbols()[0]["symbol"]
        resp = requests.get(self.base_endpoint + self.candlestick_endpoint.format(symbol, "1h"))
        print(resp.json())

        

biwrap = BinanceUsWrapper()
biwrap.get_symbol_hist()

    