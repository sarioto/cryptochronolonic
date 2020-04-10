import time
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta, datetime
import os
from statistics import mode

class FtxWrapper(object):

    base_url = "https://ftx.com/api"
    
    def __init__(self):
        return

    def get_markets(self):
        response = requests.get(self.base_url + "/markets")
        for m in response.json()["result"]:
            if m["baseCurrency"] != None and m["baseCurrency"][-4:] in ["BULL", "BEAR"] and m["name"].split("/")[-1] == "USD":
                self.get_historical(m["name"])

    def get_historical(self, mrkt_name = "XTZBULL/USD"):
        test_params = "/markets/{market_name}/candles?resolution={resolution}&limit={limit}".format(
            market_name = mrkt_name, resolution=3600, limit=10000
        )
        response = requests.get(self.base_url + test_params)
        print(len(response.json()["result"]))


ftx = FtxWrapper()
ftx.get_markets()
    