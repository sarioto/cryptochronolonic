import matplotlib.pyplot as plt
import pandas as pd  
from datetime import datetime
def pull_in_file(f_name):
    return pd.read_csv('./trade_hists/'+f_name)


if __name__ == '__main__':
<<<<<<< HEAD
    thist = pull_in_file('kraken/88_hist.txt')
=======
    thist = pull_in_file('4190_hist.txt')
>>>>>>> bbb5e3575f5c6ccc439630bc7d4f5954140a5248
    print(list(thist))
    print(thist)
    thist['date'] = thist['date'].str[:-6]
    plt.plot(thist['date'], thist['current_balance '])
    plt.xticks(rotation=90)
    plt.show()
