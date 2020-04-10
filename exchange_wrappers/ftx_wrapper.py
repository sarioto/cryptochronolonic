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
        his_data = response.json()["result"]
        df = pd.DataFrame(his_data)
        file_name = mrkt_name.split("/")[0] + "_" + mrkt_name.split("/")[-1]
        df.to_csv("./hist_data/ftx/" + file_name + ".txt")
        print("saved " + mrkt_name + " hist data")


ftx = FtxWrapper()
ftx.get_markets()
    