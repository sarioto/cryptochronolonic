import matplotlib.pyplot as plt
import pandas as pd  
from datetime import datetime
def pull_in_file(f_name):
    return pd.read_csv('./champs_histd3/'+f_name)


if __name__ == '__main__':
    thist = pull_in_file('trade_hist2633.txt')
    print(list(thist))
    print(thist.head())
    print(datetime.utcfromtimestamp(thist['date'][0]).strftime('%Y-%m-%d %H:%M:%S'))
    print(datetime.utcfromtimestamp(thist['date'][len(thist['date'])-1]).strftime('%Y-%m-%d %H:%M:%S'))
    plt.plot(thist['date'], thist['current_balance '])
    plt.show()
