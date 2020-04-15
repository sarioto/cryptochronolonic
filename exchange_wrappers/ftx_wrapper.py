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
        usd_ls = []
        for m in response.json()["result"]:
            if m["baseCurrency"] != None and m["baseCurrency"][-4:] in ["BULL", "BEAR"] and m["name"].split("/")[-1] == "USD":
                usd_ls.append(m["name"])
        return usd_ls

    def get_markets_hist(self):
        m_names = self.get_markets()
        for n in m_names:
            self.get_historical(x)
        return

    def load_hist_files_for_list(self, histFiles):
        df_dict = {}
        len_list = {}
        for sym in histFiles:
            sym_base = sym.split("_")[0][:-4]
            df = pd.read_csv("./hist_data/ftx/" +sym)
            df = self.apply_features(df)
            if (len(df) < 3000):
                continue
            if("BULL" in sym):
                df_dict["BULL"] = df
            if("BEAR" in sym):
                df_dict["BEAR"] = df
        return df_dict

    def load_all_hist_files(self, hist_list):
        df_dict = {}
        len_list = {}
        for sym in hist_list:
            base_sym = sym.split("_")[0][:-4]
            df = pd.read_csv("./hist_data/ftx/"+sym)
            df = self.apply_features(df)
            if ("BULL" in sym):
                df_dict[base_sym]["BULL"] = df
            if ("BEAR" in sym):
                df_dict[base_sym]["BEAR"] = df
        return df_dict

    def apply_features(self, df):
        df['std_close'] = df['close']/df['high']
        df['std_low'] = df['low']/df['high']
        df['std_open'] = df['open']/df['high']
        df['avg_vol_3'] = pd.Series(np.where(df.volume.rolling(34).mean() > df.volume, 1, -1), df.index)
        df["roc_close_mid"] = df["close"].pct_change(periods=34)
        df["roc_close_short"] = df["close"].pct_change(periods=13)
        df["roc_close_daily"] = df["close"].pct_change(periods=1)
        df["roc_close_long"] = df["close"].pct_change(periods=144)
        df.dropna(inplace=True)
        self.start_idx = df.index[0]
        return df

    def load_hist_files(self):
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), "../hist_data/ftx"))
        data = self.load_hist_files_for_list(histFiles)
        return data

    def load_single_df(self, base_sym):
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), "../hist_data/ftx"))
        syms = [s for s in histFiles if s.split("_")[0][:-4] == base_sym]
        print(syms)
        dfs = self.load_hist_files_for_list(syms)
        return dfs
        
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

    def get_train_frames(self, restrict_val = 0, feature_columns = ['std_close', 'std_low', 'std_open', 'avg_vol_3', "roc_close_short", "roc_close_mid", "roc_close_long", "roc_close_daily"]):
        df_dict = self.load_single_df("ALT")
        coin_and_hist_index = 0
        currentHists = df_dict
        hist_shaped = {}
        coin_dict = {}
        prefixes = []
        for ix in df_dict:
            df = currentHists[ix][feature_columns].copy()
            hist_full_size = len(df)
            as_array=np.array(df)
            hist_shaped[coin_and_hist_index] = as_array
            coin_dict[coin_and_hist_index] = ix
            coin_and_hist_index += 1
        hist_shaped = pd.Series(hist_shaped)
        return coin_dict, currentHists, hist_shaped, hist_full_size

    