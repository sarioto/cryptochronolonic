import time
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta, datetime
import os
from statistics import mode

class RhWrapper(object):

    base_url = "https://ftx.com/api"
    
    start_idxs = {}

    def __init__(self):
        return

    def get_markets(self):
        response = requests.get(self.base_url + "/markets")
        usd_ls = []
        for m in response.json()["result"]:
            if m["baseCurrency"] != None and m["baseCurrency"][-4:] in ["BULL", "BEAR"] and m["name"].split("/")[-1] == "USD":
                usd_ls.append(m["name"])
        return usd_ls

    def get_last_price(self, sym):
        response = requests.get(self.base_url + "/markets")
        for m in response.json()["result"]:
            if m["baseCurrency"] != None and m["baseCurrency"] == sym and m["name"].split("/")[-1] == "USD":
                return m["ask"]
        return

    def get_markets_hist(self, bar_limit = -1, live=False):
        m_names = self.get_markets()
        for n in m_names:
            if bar_limit != -1:
                self.get_historical(n, bar_limit, live)
            else:
                self.get_historical(n)
        return

    def load_hist_files_for_list(self, histFiles):
        df_dict = {}
        len_list = {}
        for sym in histFiles:
            sym_base = sym.split("_")[0][:-4]
            df = pd.read_csv("./hist_data/robinhood_train/" +sym)
            df = self.apply_features(df)
            if (len(df) < 3000):
                continue
            if("SPXL" == sym):
                df_dict["BULL"] = df
            if("SPXS" == sym):
                df_dict["BEAR"] = df
        return df_dict

    def load_all_hist_files(self, hist_list, live=False):
        df_dict = {}
        len_list = {}
        for sym in hist_list:
            base_sym = sym.split("_")[0][:-4]
            if base_sym not in df_dict.keys():
                df_dict[base_sym] = {}
            sym_type = sym.split("_")[0][-4:]
            if live == False:
                df = pd.read_csv("./hist_data/robinhood_train/"+sym)
            if live == True:
                df = pd.read_csv("./live_data/robinhood_train/"+sym)
            df = self.apply_features(df)
            if ("BULL" == sym_type):
                df_dict[base_sym]["BULL"] = df
            if ("BEAR" == sym_type):
                df_dict[base_sym]["BEAR"] = df
        return df_dict

    def get_matching_dataframes(self, live=False):
        data = self.load_hist_files(live)
        new_dict = {}
        for s in data:
            if live == False:
                if len(data[s]["BULL"]) == len(data[s]["BEAR"]) and s not in ("", "USDT"):
                    print(s)
                    new_dict[s] = data[s]
            else:
                if len(data[s]["BULL"]) == len(data[s]["BEAR"]) and s not in ("", "USDT"):
                    new_dict[s] = data[s]
        return new_dict

    def apply_features(self, df):
        df['std_close'] = df['close_price']/df['high_price']
        df['std_low'] = df['low_price']/df['high_price']
        df['std_open'] = df['open_price']/df['high_price']
        df['avg_vol_3'] = pd.Series(np.where(df.volume.rolling(34).mean() > df.volume, 1, -1), df.index)
        df["roc_close_mid"] = df["close_price"].pct_change(periods=34)
        df["roc_close_short"] = df["close_price"].pct_change(periods=13)
        df["roc_close_daily"] = df["close_price"].pct_change(periods=1)
        df["roc_close_long"] = df["close_price"].pct_change(periods=144)
        df.dropna(inplace=True)
        self.start_idx = df.index[0]
        return df

    def load_hist_files(self, live=False):
        if live == False:
            histFiles = os.listdir(os.path.join(os.path.dirname(__file__), "../hist_data/robinhood_train"))
        else:
            histFiles = os.listdir(os.path.join(os.path.dirname(__file__), "../live_data/robinhood_train"))
        data = self.load_all_hist_files(histFiles, live)
        return data
        
        

    def load_single_df(self, base_sym):
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), "../hist_data/ftx"))
        syms = [s for s in histFiles if s.split("_")[0][:-4] == base_sym]
        print(syms)
        dfs = self.load_hist_files_for_list(syms)
        return dfs
        
    def get_historical(self, mrkt_name = "XTZBULL/USD", bar_limit = 10000, live=False):
        test_params = "/markets/{market_name}/candles?resolution={resolution}&limit={limit}".format(
            market_name = mrkt_name, resolution=3600, limit=bar_limit
        )
        response = requests.get(self.base_url + test_params)
        his_data = response.json()["result"]
        df = pd.DataFrame(his_data)
        file_name = mrkt_name.split("/")[0] + "_" + mrkt_name.split("/")[-1]
        if live == False:
            df.to_csv("./hist_data/ftx/" + file_name + ".txt")
        else:
            df.to_csv("./live_data/ftx/" + file_name + ".txt")
        print("saved " + mrkt_name + " hist data")

    def get_train_frames_single_sym(self, restrict_val = 0, feature_columns = ['std_close', 'std_low', 'std_open', 'avg_vol_3', "roc_close_short", "roc_close_mid", "roc_close_long", "roc_close_daily"]):
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

    def get_live_frames_all_syms(self, restrict_val = 0, feature_columns = ['std_close', 'std_low', 'std_open', 'avg_vol_3', "roc_close_short", "roc_close_mid", "roc_close_long", "roc_close_daily"]):
        self.get_markets_hist(bar_limit=350, live=True)
        df_dict = self.get_matching_dataframes(live=True)
        coin_and_hist_index = 0
        currentHists = df_dict
        hist_shaped = {}
        coin_dict = {}
        prefixes = []
        hist_lengths = {}
        for s in df_dict:
            df_bull = currentHists[s]["BULL"][feature_columns].copy()
            df_bear = currentHists[s]["BEAR"][feature_columns].copy()
            self.start_idxs[s] = df_bull.index[0]
            hist_lengths[s] = len(df_bull)
            as_array_bull = np.array(df_bull)
            as_array_bear = np.array(df_bear)
            hist_shaped[coin_and_hist_index] = as_array_bull
            coin_dict[s] = coin_and_hist_index
            coin_and_hist_index += 1
            hist_shaped[coin_and_hist_index] = as_array_bear
            coin_and_hist_index += 1
        hist_shaped = pd.Series(hist_shaped)
        return coin_dict, currentHists, hist_shaped, hist_lengths

    def get_train_frames_all_syms(self, restrict_val = 0, feature_columns = ['std_close', 'std_low', 'std_open', 'avg_vol_3', "roc_close_short", "roc_close_mid", "roc_close_long", "roc_close_daily"]):
        df_dict = self.get_matching_dataframes()
        coin_and_hist_index = 0
        currentHists = df_dict
        hist_shaped = {}
        coin_dict = {}
        self.prefixes = []
        hist_lengths = {}
        for s in df_dict:
            self.prefixes.append(s)
            df_bull = currentHists[s]["BULL"][feature_columns].copy()
            df_bear = currentHists[s]["BEAR"][feature_columns].copy()
            self.start_idxs[s] = df_bull.index[0]
            hist_lengths[s] = len(df_bull)
            as_array_bull = np.array(df_bull)
            as_array_bear = np.array(df_bear)
            hist_shaped[coin_and_hist_index] = as_array_bull
            coin_dict[s] = coin_and_hist_index
            coin_and_hist_index += 1
            hist_shaped[coin_and_hist_index] = as_array_bear
            coin_and_hist_index += 1
        hist_shaped = pd.Series(hist_shaped)
        return coin_dict, currentHists, hist_shaped, hist_lengths

