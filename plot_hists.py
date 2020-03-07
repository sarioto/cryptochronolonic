import matplotlib.pyplot as plt
import pandas as pd  
from datetime import datetime
def pull_in_file(f_name):
    return pd.read_csv('./trade_hists/'+f_name)


if __name__ == '__main__':
    thist = pull_in_file('binance_per_symbol/0_hist.txt')
    plt.plot(thist['current_balance'])
    plt.xticks(rotation=90)
    plt.show()
