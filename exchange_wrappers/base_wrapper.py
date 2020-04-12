import pickle
import time
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta, datetime
import os
from statistics import mode

class BaseApiWrapper(object):
    endpoints = {
        "trade_assets": "https://api.kraken.com/0/public/AssetPairs",
        "ohlc_bars": "https://api.kraken.com/0/public/OHLC?pair={0}&since={1}&interval={2}"
    }

    his_dir = "./hists/"

    def __init__(self, hist_directory, file_path):
        self.hist_dir = hist_directory
        self.endpoints["files_path"] = file_path
        return

    def get_hist_files(self):
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), 'histories'))
        return histFiles
        
    def get_live_files(self):
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), 'paper'))
        return histFiles

    def get_usd_files(self):
        histfiles = os.listdir(os.path.join(os.path.dirname(__file__), 'usd_histories'))
        return histfiles

    # depending on the pair this may need to be overridden
    def get_file_symbol(self, sym_full):
        stripped = sym_full.split(".")[0][:-3]
        return stripped

    def load_hist_files(self, live=False):
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), self.hist_dir))
        df_dict = {}
        for sym in histFiles:
            frame = pd.DataFrame().from_csv(self.endpoints["files_path"]+sym)
            df_dict[sym] = frame
        return df_dict


    def get_train_frames(self, restrict_val = 0, feature_columns = ['vol_feat', 'std_high', 'std_close', 'roc_13'], live=False):
        df_dict = self.load_hist_files(live=live)
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
        print(mode_len)
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
                vollist.append(df['vol'][0])
        if restrict_val != 0:
            vollist = np.argsort(vollist)[-restrict_val:][::-1]
        vollist = np.argsort(vollist)[::-1]
        for ix in vollist:
            print(prefixes[ix])
            #df['vol'] = (df['vol'] - df['vol'].mean())/(df['vol'].max() - df['vol'].min())
            df = currentHists[prefixes[ix]][feature_columns].copy()
            #norm_df = (df - df.mean()) / (df.max() - df.min())
            as_array=np.array(df)
            hist_shaped[coin_and_hist_index] = as_array
            coin_dict[coin_and_hist_index] = prefixes[ix]
            coin_and_hist_index += 1
        hist_shaped = pd.Series(hist_shaped)
        return coin_dict, currentHists, hist_shaped, hist_full_size

