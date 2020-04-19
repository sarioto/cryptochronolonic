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
class HistWorker(object):
    look_back = 0
    def __init__(self, exchange_wrapper):
        self.currentHists = {}
        self.hist_shaped = {}
        self.coin_dict = {}
        self.wrapper = exchange_wrapper
        return

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
    def get_wrapper_train_frames(self, num_symbols=0):
        self.coin_dict, self.currentHists, self.hist_shaped, self.hist_full_size = self.wrapper.get_train_frames(num_symbols)
        return

    def get_wrapper_train_frames_all_syms(self, num_symbols=0):
        self.coin_dict, self.currentHists, self.hist_shaped, self.hist_sizes = self.wrapper.get_train_frames_all_syms(num_symbols)
        return

    def get_wrapper_live_frames_all_syms(self, num_symbols=0):
        self.coin_dict, self.currentHists, self.hist_shaped, self.hist_sizes = self.wrapper.get_live_frames_all_syms(num_symbols)
        return

    def get_binance_live(self, num_symbols=0):
        self.binance.fetch_usd_histories(live=True)
        self.coin_dict, self.currentHists, self.hist_shaped, self.hist_full_size = self.binance.get_train_frames(num_symbols, live=True)