from yahoo_finance import Share
import pandas as pd
import numpy as np
import os
from statistics import mode

class YahooWrapper(object):

    feature_list = ['avg_vol', 'avg_close_13', 'avg_close_21', 'avg_close_55', 'std_close', 'std_high', 'volume']

    def __init__(self, sym_list):
        self.sym_list = sym_list
        return

    def pull_historicals_three_years(self):
        self.hist_dict = {}
        for s in self.sym_list:
            yah = Share(s)


yw = YahooWrapper(['VIX'])
yw.pull_historicals_three_years()
        