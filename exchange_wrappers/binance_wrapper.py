import requests
import json
import pandas as pd
import os
import numpy as np
from statistics import mode

class BinanceUsWrapper(object):

    candlestick_endpoint = "/api/v3/klines?symbol={}&interval={}&limit={}"
    base_endpoint = "https://api.binance.us"
    info_endpoint = "/api/v3/exchangeInfo"
    live_dir = "./live_data/binance/"
    train_dir = "./hist_data/binance/"
    
    def __init__(self):
        return

    def get_symbols(self):
        response = requests.get(self.base_endpoint + self.info_endpoint)
        return response.json()["symbols"]

    def get_file_symbol(self, sym_full):
        stripped = sym_full.split(".")[0][:-3]
        return stripped

    def get_symbol_hist(self, symbol):
        print("fetching data for: ", symbol)
        resp = requests.get(self.base_endpoint + self.candlestick_endpoint.format(symbol, "4h", 800))
        return resp.json()
    
    def get_usd_symbols(self):
        syms = self.get_symbols()
        usds = []
        for x in syms:
            if x["symbol"][-3:] == "USD":
                usds.append(x["symbol"])
        return usds

    def load_hist_files(self):
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), "../hist_data/binance"))
        df_dict = {}
        for sym in histFiles:
            frame = pd.read_csv("./hist_data/binance/" +sym)
            df_dict[sym] = frame
        return df_dict
    
    def load_live_files(self):
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), "../hist_data/binance"))
        df_dict = {}
        for sym in histFiles:
            frame = pd.read_csv("./hist_data/binance/" +sym)
            df_dict[sym] = frame
        return df_dict

    def fetch_usd_histories(self, live=False):
        syms = self.get_usd_symbols()
        store_dir = self.train_dir
        if live:
            self.live_dir
        for s in syms:
            data = self.get_symbol_hist(s)
            df = pd.DataFrame(data)
            df = df[df.columns[:6]]
            df.columns = ["date", "open", "high", "low", "close", "volume"]
            df = df.convert_objects(convert_numeric=True)
            df = df.fillna(method='ffill')
            df["hl_spread"] = df["low"] / df["high"] 
            df["oc_spread"] = df["close"] / df["open"]
            df["roc_close"] = df["close"].pct_change(periods=8)
            df["roc_volume"] = df["volume"].pct_change(periods=16)
            #df["rolling_spread"] = df["oc_spread"].rolling(34).mean() / df["oc_spread"]
            df["volume_feature"] = pd.Series(np.where(df.volume.rolling(3).mean() > df.volume, 1, -1), df.index)
            '''
            df['vol_feat'] = df.vol.rolling(3).mean() / df.vol
            df['avg_close_3'] = pd.Series(np.where(df.close.rolling(3).mean() / df.close, 1, -1),df.index)
            df['avg_close_13'] = pd.Series(np.where(df.close.rolling(21).mean() / df.close.rolling(3).mean(), 1, -1),df.index)
            df["roc_13"] = df.close.pct_change(periods=13)
            df['std_close'] = df['open']/df['close']
            df['std_high'] = df['low']/df['close']

            df['std_close'] = df['close']/df['high']
            df['std_high'] = df['high']/df['high']
            df['std_low'] = df['low']/df['high']
            df['std_open'] = df['open']/df['high']
            df['avg_vol_3'] = pd.Series(np.where(df.volume.rolling(3).mean() > df.volume, 1, 0), df.index)
            '''
            df.dropna(inplace=True)
            df = df.iloc[::-1].reset_index()
            if len(df) > 0:
                df.to_csv("./hist_data/binance/" + s + ".txt")

    def get_train_frames(self, restrict_val = 0, feature_columns = ['hl_spread', 'oc_spread', 'roc_close', 'roc_volume', 'volume_feature']):
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

