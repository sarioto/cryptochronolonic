#from yahoo_finance import Share
import robin_stocks as r
import pandas as pd
import numpy as np
import os

class RobinHoodWrapper(object):

    feature_list = ['avg_vol', 'avg_close_13', 'avg_close_21', 'avg_close_55', 'std_close', 'std_high', 'volume', 'begins_at']
    def __init__(self, lookback = 55):
        self.lb = lookback
        self.api_init() 
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
        print(creds)
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
    
    def get_spxl_spxs_hist(self, tf="5year"):
        df_dict = {}
        results = r.get_historicals("SPXL", span=tf)
        df = pd.DataFrame().from_dict(results)
        df_dict["SPXL"] = df
        results = r.get_historicals("SPXS", span=tf)
        df = pd.DataFrame().from_dict(results)
        df_dict["SPXS"] = df
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
            frame.to_csv("../hist_data/robinhood_train/"+x+".txt")
        return

    def load_train_data(self, restrict_val = 0):
        fileNames = self.get_train_filenames()
        coin_and_hist_index = 0
        file_lens = []
        currentHists = {}
        for y in range(0,len(fileNames)):
            df = self.load_df_from_file(fileNames[y])
            df_len = len(df)
            #print(df.head())
            file_lens.append(df_len)
        mode_len = mode(file_lens)
        print(mode_len)
        hist_full_size = mode_len
        vollist = []
        prefixes = []
        for x in range(0, len(fileNames)):
            df = self.load_df_from_file(fileNames[x])
            as_array = np.array(df)
            col_prefix = self.get_file_symbol(fileNames[x])
            #as_array = np.array(df)
            if(len(as_array) == mode_len):
                #print(as_array)
                prefixes.append(col_prefix)
                currentHists[col_prefix] = df
                vollist.append(df['volume'][0])
        if restrict_val != 0:
            vollist = np.argsort(vollist)[-restrict_val:][::-1]
        vollist = np.argsort(vollist)[::-1]
        #print(vollist)
        for ix in vollist:
            print(prefixes[ix])
            df['volume'] = (df['volume'] - df['volume'].mean())/(df['volume'].max() - df['volume'].min())
            df = self.currentHists[prefixes[ix]][['volume', 'std_high', 'std_close', 'avg_vol_3', 'avg_close_3', 'avg_close_13', 'avg_close_34']].copy()
            #norm_df = (df - df.mean()) / (df.max() - df.min())
            as_array=np.array(df)
            self.hist_shaped[coin_and_hist_index] = as_array
            self.coin_dict[coin_and_hist_index] = prefixes[ix]
            coin_and_hist_index += 1
        self.hist_shaped = pd.Series(self.hist_shaped)



rw = RobinHoodWrapper()
print(rw.get_train_filenames())