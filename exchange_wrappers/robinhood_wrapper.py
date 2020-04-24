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

    def load_single_df(self, base_sym):
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), "../robinhhood_train/ftx"))
        syms = [s for s in histFiles if s.split("_")[0][:-4] == base_sym]
        print(syms)
        dfs = self.load_hist_files_for_list(syms)
        return dfs

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

    def load_train_data(self, restrict_val = 0):
        fileNames = self.get_train_filenames()
        coin_and_hist_index = 0
        file_lens = []
        currentHists = {}
        hist_shaped = {}
        coin_dict = {}
        vollist = []
        prefixes = []
        hist_full_sized = 0
        for x in range(0, len(fileNames)):
            df = self.load_df_from_file("./hist_data/robinhood_train/"+fileNames[x])
            as_array = np.array(df)
            col_prefix = self.get_file_symbol(fileNames[x])
            #print(as_array)
            prefixes.append(col_prefix)
            currentHists[col_prefix] = df
            hist_full_sized = len(df)
            print(len(df))
        #print(vollist)
        for s in currentHists:
            if(hist_full_sized > len(currentHists[s])):
                print("trimming df")
                currentHists[s].drop(currentHists[s].tail(hist_full_sized-len(currentHists[s])).index, inplace=True)
        for ix in range(0,len(prefixes)):
            #print(prefixes[ix])
            df = currentHists[prefixes[ix]]
            df = df[self.feature_list].copy()
            #norm_df = (df - df.mean()) / (df.max() - df.min())
            as_array=np.array(df)
            #print(as_array)
            hist_shaped[ix] = as_array
            coin_dict[ix] = prefixes[ix]
        hist_shaped = pd.Series(hist_shaped)
        #print(hist_shaped[0][0])
        #print(hist_shaped[0][1])
        return coin_dict, currentHists, hist_shaped, hist_full_sized

rh = RobinHoodWrapper()
rh.get_spxl_spxs_hist()
#print(rh.load_train_data()[1]["SPXL"])
