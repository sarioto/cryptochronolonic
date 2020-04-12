import pickle
import time
import pandas as pd
import requests
import numpy as np
#from poloniex import Poloniex
from binance.client import Client
from datetime import date, timedelta, datetime
import os
from statistics import mode
from exchange_wrappers import kraken_wrapper, binance_wrapper
#from ephemGravityWrapper import gravity as gbaby
'''
As can be expected by this point, you will notice that
nothing done here has been done in the best possible way
so feel STRONGLY ENCOURAGED TO FORK AND FIX/UPDATE/REFACTOR,
also for the sake of not running computations for computations
sake instead of calculating the actual gravitational pull we
will just tack on a column of moon distances since its porportional
to the gravitational pull
and occurs at the same intervals
'''

'''
the properties for histworker are set for the most part in combine frames which is called from the constructor
'''
class HistWorker:
    look_back = 0
    def __init__(self):
        self.currentHists = {}
        self.hist_shaped = {}
        self.coin_dict = {}
        #self.combine_frames()
        self.look_back = 90
        self.hist_full_size = self.look_back * 12
        #self.binance_client = Client("", "")
        self.kw = kraken_wrapper.KrakenWrapper()
        self.binance = binance_wrapper.BinanceUsWrapper()
        #self.rh = robinhood_wrapper.RobinHoodWrapper()
        return

    def get_hist_files(self):
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), 'histories'))
        return histFiles

    def get_binance_hist_files(self):
        binanceFiles = os.listdir(os.path.join(os.path.dirname(__file__), 'binance_hist'))
        return binanceFiles

    def get_live_files(self):
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), 'paper'))
        return histFiles

    def get_usd_files(self):
        histfiles = os.listdir(os.path.join(os.path.dirname(__file__), 'usd_histories'))
        return histfiles

    def get_gdax_training_files(self):
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), '../gdax'))
        return histFiles

    def get_data_frame(self, fname):
        frame = pd.read_csv('./histories/'+fname) # timestamps will but used as index
        return frame
    def get_binance_frames(self, fname):
        frame = pd.read_csv('./binance_hist/'+fname) # timestamps will but used as index
        return frame

    def get_live_data_frame(self, fname):
        frame = pd.read_csv('./paper/'+fname)
        return frame

    def get_file_as_frame(self, fname):
        frame = pd.read_csv('../gdax/'+fname)
        return frame
    
    def get_polo_usd_frame(self, fname):
        return pd.read_csv("./usd_histories/"+fname)

    def get_polo_usd_live_frame(self, fname):
        return pd.read_csv("./usd_live/"+fname)

    def get_file_symbol(self, f):
        f = f.split("_", 2)
        return f[1]
    
    def get_usdt_file_symbol(self, f):
        f = f.split("_", 2)
        return f[1]

    def get_binance_symbol(self, f):
        f = f.split("_", 2)
        return f[0]

    def get_kraken_syms(self):
        self.kw.get_assets()

    '''
    def get_data_for_astro(self):
        data = {}
        dates = []
        md = []
        l_of_frame = len(self.currentHists['DASH']['date'])
        for snoz in range(0, l_of_frame):
            new_date = datetime.utcfromtimestamp((self.currentHists['DASH']['date'][snoz])).strftime('%Y-%m-%d  %H:%M:%S')
            dates.append(self.currentHists['DASH']['date'][snoz])
            md.append(gbaby.planet_dist(new_date, 'moon'))

        data = {'date': dates, 'moon_dist': md}
        data = pd.DataFrame.from_dict(data)
        data.to_csv("moon_dists.txt", encoding="utf-8")
        #data = data.join(self.currentHists['DASH'].set_index('date'), on='date', how="left").drop('Unnamed: 0', 1)
        return data.head()
    '''
    def read_in_moon_data(self, df):
        moon = pd.read_csv('./moon_dists.txt')
        moon.set_index("date")
        moon.drop("Unnamed: 0", 1)
        df = df.drop('Unnamed: 0', 1).set_index("date")
        return moon.join(df, on="date")

    #BEGIN BINANCE METHODS

    def pull_binance_symbols(self):
        sym_list = []
        for x in self.binance_client.get_products()["data"]:
            sym_list.append(x["symbol"])
        return sym_list

    def get_binance_hist_frame(self, symbol):
        frame = hs.binance_client.get_historical_klines(symbol, Client.KLINE_INTERVAL_30MINUTE, "1 May, 2018", "1 Jan, 2019")
        for x in range(len(frame)):
            frame[x] = frame[x][:6]
        frame = pd.DataFrame(frame, columns=["date", "open", "high", "low", "close", "volume"])
        return frame

    def write_binance_training_files(self, syms):
        for s in range(len(syms)):
            frame = self.get_binance_hist_frame(syms[s])
            frame['avg_vol_3'] = frame['volume'].rolling(3).mean()
            frame['avg_vol_13'] = frame['volume'].rolling(13).mean()
            frame['avg_vol_34'] = frame['volume'].rolling(34).mean()
            frame['avg_close_3'] = frame['close'].rolling(3).mean()
            frame['avg_close_13'] = frame['close'].rolling(13).mean()
            frame['avg_close_34'] = frame['close'].rolling(34).mean()
            frame.fillna(value=-99999, inplace=True)
            frame.to_csv("./binance_hist/"+syms[s]+"_hist.txt", encoding="utf-8")

    def combine_binance_frames_vol_sorted(self, restrict_val=0):
        fileNames = self.get_binance_hist_files()
        coin_and_hist_index = 0
        file_lens = []
        for y in range(0,len(fileNames)):
            df = self.get_binance_frames(fileNames[y])
            df_len = len(df)
            #print(df.head())
            file_lens.append(df_len)
        mode_len = mode(file_lens)
        print(mode_len)
        vollist = []
        prefixes = []
        for x in range(0, len(fileNames)):
            df = self.get_binance_frames(fileNames[x])
            col_prefix = self.get_binance_symbol(fileNames[x])
            as_array = np.array(df)
            if(len(as_array) == mode_len and col_prefix[-3:] == "BTC"):
                #print(as_array)
                prefixes.append(col_prefix)
                self.currentHists[col_prefix] = df
                vollist.append(df['volume'][0])
        if restrict_val != 0:
            vollist = np.argsort(vollist)[-restrict_val:][::-1]
        vollist = np.argsort(vollist)[::-1]
        for ix in vollist:
            df = self.currentHists[prefixes[ix]].copy()
            norm_df = (df - df.mean()) / (df.max() - df.min())
            as_array=np.array(norm_df)
            self.hist_shaped[coin_and_hist_index] = as_array
            self.coin_dict[coin_and_hist_index] = prefixes[ix]
            coin_and_hist_index += 1
        self.hist_shaped = pd.Series(self.hist_shaped)
        print(self.currentHists.keys(), self.coin_dict)

    def combine_binance_frames(self):
        fileNames = self.get_binance_hist_files()
        coin_and_hist_index = 0
        file_lens = []
        for y in range(0,len(fileNames)):
            df = self.get_binance_frames(fileNames[y])
            df_len = len(df)
            #print(df.head())
            file_lens.append(df_len)
        mode_len = mode(file_lens)
        print(mode_len)
        for x in range(0, len(fileNames)):
            df = self.get_binance_frames(fileNames[y])
            col_prefix = self.get_binance_symbol(fileNames[x])
            #df.drop("Unnamed: 0", 1)
            #df = self.read_in_moon_data(df)
            #df = df.drop("Unnamed: 0", 1)
            #df.rename(columns = lambda x: col_prefix+'_'+x, inplace=True)
            as_array = np.array(df)

            #print(len(as_array))
            if(len(as_array) == mode_len and col_prefix[-3:] == "BTC"):
                #print(as_array)
                self.currentHists[col_prefix] = df.copy()
                #print(self.currentHists[col_prefix].head())
                norm_df = (df - df.mean()) / (df.max() - df.min())
                as_array=np.array(norm_df)
                self.hist_shaped[coin_and_hist_index] = as_array
                self.coin_dict[coin_and_hist_index] = col_prefix
                coin_and_hist_index += 1
        self.hist_shaped = pd.Series(self.hist_shaped)

    def combine_frames(self):
        length = 7992
        fileNames = self.get_hist_files()
        coin_and_hist_index = 0
        for x in range(0,len(fileNames)):
            df = self.get_data_frame(fileNames[x])
            col_prefix = self.get_file_symbol(fileNames[x])
            #df.drop("Unnamed: 0", 1)
            #df = self.read_in_moon_data(df)
            df = df.drop("Unnamed: 0", 1)
            #df.rename(columns = lambda x: col_prefix+'_'+x, inplace=True)

            as_array = np.array(df)

            #print(len(as_array))
            if(len(as_array) == length):
                self.currentHists[col_prefix] = df
                df = (df - df.mean()) / (df.max() - df.min())
                as_array=np.array(df)
                self.hist_shaped[coin_and_hist_index] = as_array
                self.coin_dict[coin_and_hist_index] = col_prefix
                coin_and_hist_index += 1
        self.hist_shaped = pd.Series(self.hist_shaped)
        '''
        main = df_list[0]
        for i in range(1, len(df_list)):
            main = main.join(df_list[i])
        return main
        '''
    def combine_polo_frames_vol_sorted(self, restrict_val=0):
        length = 7992
        fileNames = self.get_hist_files()
        coin_and_hist_index = 0
        file_lens = []
        for y in range(0,len(fileNames)):
            df = self.get_data_frame(fileNames[y])
            df_len = len(df)
            #print(df.head())
            file_lens.append(df_len)
        mode_len = mode(file_lens)
        print(mode_len)
        self.hist_full_size = mode_len
        vollist = []
        prefixes = []
        for x in range(0, len(fileNames)):
            df = self.get_data_frame(fileNames[x])
            col_prefix = self.get_file_symbol(fileNames[x])
            as_array = np.array(df)
            if(len(as_array) == mode_len):
                #print(as_array)
                prefixes.append(col_prefix)
                self.currentHists[col_prefix] = df
                vollist.append(df['volume'][0])
        if restrict_val != 0:
            vollist = np.argsort(vollist)[-restrict_val:][::-1]
        vollist = np.argsort(vollist)[::-1]
        #print(vollist)
        for ix in vollist:
            print(prefixes[ix])
            #print(self.currentHists[col_prefix].head())
            df = self.currentHists[prefixes[ix]][['std_high', 'std_close', 'std_open', 'avg_vol_3', 'avg_close_3', 'avg_close_13', 'avg_close_34']].copy()
            #norm_df = (df - df.mean()) / (df.max() - df.min())
            as_array=np.array(df)
            self.hist_shaped[coin_and_hist_index] = as_array
            self.coin_dict[coin_and_hist_index] = prefixes[ix]
            coin_and_hist_index += 1
        self.hist_shaped = pd.Series(self.hist_shaped)


    def combine_live_frames(self, length, base_sym=""):
        fileNames = self.get_live_files()
        coin_and_hist_index = 0
        file_lens = []
        for y in range(0,len(fileNames)):
            df = self.get_live_data_frame(fileNames[y])
            df_len = len(df)
            #print(len(df))
            file_lens.append(df_len)
        mode_len = mode(file_lens)
        self.hist_full_size = mode_len
        for x in range(0,len(fileNames)):
            df = self.get_live_data_frame(fileNames[x])
            col_prefix = self.get_file_symbol(fileNames[x])
            #df.drop("Unnamed: 0", 1)
            #df = self.read_in_moon_data(df)
            #df = df.drop("Unnamed: 0", 1)
            #df.rename(columns = lambda x: col_prefix+'_'+x, inplace=True)
            as_array = np.array(df)
            #print(len(as_array))
            #print(len(as_array))
            if(len(as_array) == mode_len):
                #print("adding df")
                self.currentHists[col_prefix] = df
                df = (df - df.mean()) / (df.max() - df.min())
                as_array = np.array(df)
                self.hist_shaped[coin_and_hist_index] = as_array
                self.coin_dict[coin_and_hist_index] = col_prefix
                coin_and_hist_index += 1
        #print(self.hist_shaped)
        self.hist_shaped = pd.Series(self.hist_shaped)
        '''
        main = df_list[0]
        for i in range(1, len(df_list)):
            main = main.join(df_list[i])
        return main
        '''
    #TODO implement the same logic as is used in the polo_frames_sorted combination
    def combine_polo_usd_frames(self, restrict_val = 0):
        fileNames = self.get_usd_files()
        coin_and_hist_index = 0
        file_lens = []
        for y in range(0,len(fileNames)):
            df = self.get_polo_usd_frame(fileNames[y])
            df_len = len(df)
            #print(df.head())
            file_lens.append(df_len)
        mode_len = mode(file_lens)
        print(mode_len)
        self.hist_full_size = mode_len
        vollist = []
        prefixes = []
        for x in range(0, len(fileNames)):
            df = self.get_polo_usd_frame(fileNames[x])
            as_array = np.array(df)
            col_prefix = self.get_file_symbol(fileNames[x])
            #as_array = np.array(df)
            if(len(as_array) == mode_len):
                #print(as_array)
                prefixes.append(col_prefix)
                self.currentHists[col_prefix] = df
                vollist.append(df['volume'][0])
        if restrict_val != 0:
            vollist = np.argsort(vollist)[-restrict_val:][::-1]
        vollist = np.argsort(vollist)[::-1]
        #print(vollist)
        for ix in vollist:
            print(prefixes[ix])
            df = self.currentHists[prefixes[ix]]
            df['volume'] = (df['volume'] - df['volume'].mean())/(df['volume'].max() - df['volume'].min())
            df = df[['volume', 'std_high', 'std_close', 'avg_vol_3', 'avg_close_3', 'avg_close_13', 'avg_close_34']].copy()
            #norm_df = (df - df.mean()) / (df.max() - df.min())
            as_array=np.array(df)
            self.hist_shaped[coin_and_hist_index] = as_array
            self.coin_dict[coin_and_hist_index] = prefixes[ix]
            coin_and_hist_index += 1
        self.hist_shaped = pd.Series(self.hist_shaped)

    def combine_live_usd_frames(self, restrict_val = 0):
        fileNames = self.get_usd_files()
        coin_and_hist_index = 0
        file_lens = []
        for y in range(0,len(fileNames)):
            df = self.get_polo_usd_live_frame(fileNames[y])
            df_len = len(df)
            #print(df.head())
            file_lens.append(df_len)
        mode_len = mode(file_lens)
        print(mode_len)
        self.hist_full_size = mode_len
        vollist = []
        prefixes = []
        for x in range(0, len(fileNames)):
            df = self.get_polo_usd_live_frame(fileNames[x])
            col_prefix = self.get_file_symbol(fileNames[x])
            as_array = np.array(df)
            if(len(as_array) == mode_len):
                #print(as_array)
                prefixes.append(col_prefix)
                self.currentHists[col_prefix] = df
                vollist.append(df['volume'][0])
        if restrict_val != 0:
            vollist = np.argsort(vollist)[-restrict_val:][::-1]
        vollist = np.argsort(vollist)[::-1]
        #print(vollist)
        for ix in vollist:
            print(prefixes[ix])
            #print(self.currentHists[col_prefix].head())
            df = self.currentHists[prefixes[ix]]
            df['volume'] = (df['volume'] - df['volume'].mean())/(df['volume'].max() - df['volume'].min())

            df = df[['volume', 'std_high', 'std_close', 'avg_vol_3', 'avg_close_3', 'avg_close_13', 'avg_close_34']].copy()
            #norm_df = (df - df.mean()) / (df.max() - df.min())
            as_array=np.array(df)
            self.hist_shaped[coin_and_hist_index] = as_array
            self.coin_dict[coin_and_hist_index] = prefixes[ix]
            coin_and_hist_index += 1
        self.hist_shaped = pd.Series(self.hist_shaped)

    def pull_robinhood_train_data(self):
        self.rh.get_spxl_spxs_hist()
        return

    def get_robinhood_train(self):
        self.coin_dict, self.currentHists, self.hist_shaped, self.hist_full_size = self.rh.load_train_data()
        return

    def pull_kraken_hist(self):
        self.kw.pull_kraken_hist_usd()

    def get_kraken_train(self, num_symbols=3):
        self.coin_dict, self.currentHists, self.hist_shaped, self.hist_full_size = self.kw.get_train_frames(num_symbols)
        return

    def get_binance_train(self, num_symbols=0):
        #self.binance.fetch_usd_histories()
        self.coin_dict, self.currentHists, self.hist_shaped, self.hist_full_size = self.binance.get_train_frames(num_symbols)
        return

    def get_binance_live(self, num_symbols=0):
        self.binance.fetch_usd_histories(live=True)
        self.coin_dict, self.currentHists, self.hist_shaped, self.hist_full_size = self.binance.get_train_frames(num_symbols, live=True)