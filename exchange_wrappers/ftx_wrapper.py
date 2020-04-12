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
        for sym_pairs in histFiles:
            sym, sym_two = sym_pairs[0], sym_pairs[1]
            base_sym = sym.split("_")[0][:-4]
            df = pd.read_csv("./hist_data/ftx/" +sym)
            df_two = pd.read_csv("./hist_data/fex/" + sym_two)
            if (len(df) > len(df_two)):
                df = 
            df = self.apply_features(df)
            df_two = self.apply_features(df_two)
            df_dict[base_sym] = {
                sym: df,
                sym_two: df_two
            }
        return df_dict

    def apply_features(self, df):
        df['std_close'] = df['close']/df['high']
        df['std_low'] = df['low']/df['high']
        df['std_open'] = df['open']/df['high']
        df['avg_vol_3'] = pd.Series(np.where(df.volume.rolling(3).mean() > df.volume, 1, -1), df.index)
        return df

    def load_hist_files(self):
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), "../hist_data/ftx"))
        data = self.load_hist_files_for_list(histFiles)
        return data

    def load_single_df(self, base_sym):
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), "../hist_data/ftx"))
        syms = [s for s in histFiles if s.split("_")[0][:-4] == base_sym]
        dfs = self.load_hist_files_for_list([syms])
        
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

    def get_train_frames(self, restrict_val = 0, feature_columns = ['std_close', 'std_low', 'std_open', 'avg_vol_3']):
        df_dict = self.load_hist_files()
        coin_and_hist_index = 0
        file_lens = []
        currentHists = {}
        hist_shaped = {}
        coin_dict = {}
        vollist = []
        prefixes = []
        for y in df_dict:
            df = df_dict[y]
            df_len = len(df)
            #print(df.head())
            file_lens.append(df_len)
        mode_len = mode(file_lens)
        hist_full_size = mode_len
        vollist = []
        prefixes = []
        for x in df_dict:
            df = df_dict[x]
            col_prefix = self.get_file_symbol(x)
            #as_array = np.array(df)
            if(len(df) == mode_len):
                #print(as_array)
                prefixes.append(col_prefix)
                currentHists[col_prefix] = df
                print(len(df["volume"]))
                print(col_prefix)
                vollist.append(df['volume'][0])
        if restrict_val != 0:
            vollist = np.argsort(vollist)[-restrict_val:][::-1]
        vollist = np.argsort(vollist)[::-1]
        for ix in vollist:
            #df['vol'] = (df['vol'] - df['vol'].mean())/(df['vol'].max() - df['vol'].min())
            df = currentHists[prefixes[ix]][feature_columns].copy()
            #norm_df = (df - df.mean()) / (df.max() - df.min())
            as_array=np.array(df)
            hist_shaped[coin_and_hist_index] = as_array
            coin_dict[coin_and_hist_index] = prefixes[ix]
            coin_and_hist_index += 1
        hist_shaped = pd.Series(hist_shaped)
        return coin_dict, currentHists, hist_shaped, hist_full_size

ftx = FtxWrapper()
dfs = ftx.load_single_df("ALT")
    