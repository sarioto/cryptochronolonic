import pandas as pd
import numpy as np
import os
from statistics import mode
from urllib.request import urlopen
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

class YahooWrapper(object):
    bardata_points = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    num_points = len(bardata_points)

    def __init__(self, sym):
        self.sym = sym
        return

    def get_year_historical(self):
        page = urlopen('https://finance.yahoo.com/quote/%5E'+self.sym + '/history/').read()
        soup = BeautifulSoup(page)
        table = soup.findAll("table")[0]
        data_dict = {}
        row_num = 0
        for row in table.findAll("tr")[:-1]:
            datas = row.findAll("td")
            if(len(datas) > 0):
                current_date = row_num
                data_dict[current_date] = {"date": datas[0].get_text()}
                for d in range(1, len(datas)-1):
                    point_idx = d % self.num_points
                    col_name = self.bardata_points[point_idx]
                    data_dict[current_date][col_name] = float(datas[d].get_text())
                row_num += 1
        #print(data_dict)
        df_transpose = pd.DataFrame().from_dict(data_dict).T
        df_transpose = df_transpose.sort_index()
        df_transpose = df_transpose[::-1]
        df_transpose.to_csv(self.sym + ".txt")
        return self.add_features_and_save(df_transpose)

    def add_features_and_save(self, df):
        df["rolling_fast"] = df['close'].rolling(13).mean()
        df["roc"] = df["close"].pct_change(5)
        df["ichi_eight_balls"] = (pd.rolling_max(df["high"], window=8) + pd.rolling_min(df["low"], window=8))/2
        print(df["close"])
        df.dropna(inplace=True)
        return df

    def test_slow_fast(self):
        df = self.get_year_historical()
        portfolio = {}
        portfolio["usd"] = 100000
        portfolio[self.sym] = 0
        epoch_count = len(df)
        total_usd = []
        for idx in range(epoch_count):
            #print(df.iloc[idx])
            close =  df["close"].iloc[idx]
            usd = portfolio["usd"]
            shares = portfolio[self.sym]
            total = usd + (close * shares)
            print("close: {} usd: {} shares: {} total: {}".format(close, usd, shares, total))
            total_usd.append(total)
            if(close > df["rolling_fast"].iloc[idx] and usd > close):
                portfolio["usd"] = usd - ((usd//close) * close)
                portfolio[self.sym] = shares + usd // close
                #print("bought ", portfolio[self.sym], " shares")
            if(close < df["rolling_fast"].iloc[idx] and shares > 0):
                print("selling {} for {} usd at {} price".format(shares, shares * close, close))
                portfolio["usd"] = usd + (shares * close)
                portfolio[self.sym] = 0
            #print("usd: ", portfolio["usd"], " " + self.sym + ": ", portfolio[self.sym] * close)
        self.plot_array(total_usd)

    def plot_array(self, array_in):
        plt.plot(array_in)
        plt.xticks(rotation=90)
        plt.show()

yw = YahooWrapper('VIX')
yw.test_slow_fast()
        