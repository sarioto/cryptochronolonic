import requests
import json
import pandas as pd


class BinanceUsWrapper(object):

    candlestick_endpoint = "/api/v3/klines"
    base_endpoint = "https://api.binance.us"
    info_endpoint = "/api/v3/exchangeInfo"
    def __init__(self):
        return

    def get_info(self):
        response = requests.get(self.base_endpoint + self.info_endpoint)
        print(len(response.json()["symbols"]))

biwrap = BinanceUsWrapper()
biwrap.get_info()

    