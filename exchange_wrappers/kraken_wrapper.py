import pickle
import time
import pandas as pd
import requests
from datetime import date, timedelta, datetime

class KrakenWrapper(object):
    endpoints = {
        "trade_assets": "https://api.kraken.com/0/public/AssetPairs",
        "ohlc_bars": "https://api.kraken.com/0/public/OHLC?pair={0}&since={1}&interval={2}"
    }

    def __init__(self, key="", secret="", default_lookback = 233, default_lb_interval = 240):
        self.key = key
        self.secret = secret
        self.look_back = default_lookback
        self.lb_interval = default_lb_interval
        return 

    def get_assets(self, base_pair):
        tradeable_assets = requests.get(self.endpoints["trade_assets"])
        asset_list = []
        #df = pd.DataFrame(tradeable_assets.json())
        for i in tradeable_assets.json()["result"]:
            if i[-(len(base_pair)):] == base_pair:
                asset_list.append(i)
        return asset_list

    def pull_kraken_hist_usd(self):
        sym_list = self.get_assets("USD")
        lookback_ts = datetime.today() - timedelta(self.look_back)
        for i in sym_list:
            hist_req = requests.get(self.endpoints["ohlc_bars"].format(i, lookback_ts, self.lb_interval))
            df = pd.DataFrame(hist_req.json()["result"])
            df.to_csv("./hist_data/kraken_data/" + i + ".txt")
            print("saved ", i, " to csv no col labels")