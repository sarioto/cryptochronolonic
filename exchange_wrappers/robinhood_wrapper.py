#from yahoo_finance import Share
import robin_stocks as r
import pandas as pd
import numpy as np
import os
from statistics import mode

class RobinHoodWrapper(object):
    sym_list = ["SPXL", "SPXS", "CBOE"]
    feature_list = ['avg_vol', 'avg_close_13', 'avg_close_21', 'avg_close_55', 'std_close', 'std_high', 'volume']
    def __init__(self, lookback = 55):
        print("reinitializing")
        self.lb = lookback 
        return

    def get_keys(self):
        with open("./godsplan.txt") as f:
            content = f.readlines()
            content[0] = content[0][:-1]
            if (content[1][-1:] == "\n"):
                content[1] = content[1][:-1]
            return content

    def api_init(self):
        creds = self.get_keys()
        r.login(creds[0], creds[1])
        return

    def get_train_filenames(self):
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), '../hist_data/robinhood_train/'))
        return histFiles

    def get_file_symbol(self, filename):
        sym = filename.split('.')[0]
        return sym

    def load_df_from_file(self, file_name):
        df = pd.DataFrame().from_csv(file_name)
        return df
    
    def load_hist_files_for_list(self, histFiles):
        df_dict = {}
        len_list = {}
        for sym in histFiles:
            sym_base = sym.split(".")[0]
            df = pd.read_csv("./hist_data/robinhood_train/" +sym)
            df = self.apply_features(df)
            if("SPXL" == sym_base):
                df_dict["BULL"] = df
            if("SPXS" == sym_base):
                df_dict["BEAR"] = df
        return df_dict

    def load_single_df(self):
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), "../hist_data/robinhood_train"))
        syms = [s for s in histFiles]
        print(syms)
        dfs = self.load_hist_files_for_list(syms)
        return dfs

    def get_train_frames(self, restrict_val = 0, feature_columns = ['std_close', 'std_low', 'std_open', 'avg_vol_3', "roc_close_short", "roc_close_mid", "roc_close_long", "roc_close_daily"]):
        df_dict = self.load_single_df()
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

    def get_matching_dataframes(self, live=False):
        data = self.load_hist_files(live)
        new_dict = {}
        for s in data:
            if live == False:
                if len(data[s]["BULL"]) == len(data[s]["BEAR"]) and len(data[s]["BULL"]) > 3000 and s not in ("", "USDT"):
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

    def get_spxl_spxs_hist(self, tf="year"):
        self.api_init()
        df_dict = {}
        results = r.get_historicals("SPXL", span=tf)
        df = pd.DataFrame().from_dict(results)
        df_long = df
        results = r.get_historicals("SPXS", span=tf)
        df = pd.DataFrame().from_dict(results)
        df_short = df
        '''
        for x in df_dict:
            frame = df_dict[x]
            frame["close_price"] = pd.to_numeric(frame["close_price"])
            frame["low_price"] = pd.to_numeric(frame["low_price"])
            frame["high_price"] = pd.to_numeric(frame["high_price"])
            frame["open_price"] = pd.to_numeric(frame["open_price"])
            frame["volume"] = pd.to_numeric(frame["volume"])
            frame['avg_vol'] = pd.Series(np.where(frame.volume.rolling(13).mean() / frame.volume.rolling(8).mean(), 1, 0),frame.index)
            frame['avg_close_13'] = pd.Series(np.where(frame.close_price.rolling(13).mean() / frame.close_price.rolling(3).mean(), 1, 0),frame.index)
            frame['avg_close_21'] = pd.Series(np.where(frame.close_price.rolling(21).mean() / frame.close_price.rolling(8).mean(), 1, 0),frame.index)
            frame['avg_close_55'] = pd.Series(np.where(frame.close_price.rolling(55).mean() / frame.close_price.rolling(21).mean(), 1, 0),frame.index)
            frame['std_close'] = frame['open_price']/frame['close_price']
            frame['std_high'] = frame['low_price']/frame['high_price']
            frame = frame.iloc[::-1].reset_index()
            frame.to_csv("./hist_data/robinhood_train/"+x+".txt")
        '''
        df_long.to_csv("./hist_data/robinhood_train/SPXL.txt")
        df_short.to_csv("./hist_data/robinhood_train/SPXS.txt")
        return 

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

rh = RobinHoodWrapper()
rh.get_train_frames()
#print(rh.load_train_data()[1]["SPXL"])
